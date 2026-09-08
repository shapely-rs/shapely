use facet_core::TryFromOutcome;
use facet_path::{Path, PathStep};

use super::*;
use crate::typeplan::{DeserStrategy, TypePlanNodeKind};

////////////////////////////////////////////////////////////////////////////////////////////////////
// Misc.
////////////////////////////////////////////////////////////////////////////////////////////////////
impl<'facet, const BORROW: bool> Partial<'facet, BORROW> {
    /// Applies a closure to this Partial, enabling chaining with operations that
    /// take ownership and return `Result<Self, E>`.
    ///
    /// This is useful for chaining deserializer methods that need `&mut self`:
    ///
    /// ```ignore
    /// wip = wip
    ///     .begin_field("name")?
    ///     .with(|w| deserializer.deserialize_into(w))?
    ///     .end()?;
    /// ```
    #[inline]
    pub fn with<F, E>(self, f: F) -> Result<Self, E>
    where
        F: FnOnce(Self) -> Result<Self, E>,
    {
        f(self)
    }

    /// Returns true if the Partial is in an active state (not built or poisoned).
    ///
    /// After `build()` succeeds or after an error causes poisoning, the Partial
    /// becomes inactive and most operations will fail.
    #[inline]
    pub fn is_active(&self) -> bool {
        self.state == PartialState::Active
    }

    /// Returns the current frame count (depth of nesting)
    ///
    /// The initial frame count is 1 — `begin_field` would push a new frame,
    /// bringing it to 2, then `end` would bring it back to `1`.
    ///
    /// This is an implementation detail of `Partial`, kinda, but deserializers
    /// might use this for debug assertions, to make sure the state is what
    /// they think it is.
    #[inline]
    pub const fn frame_count(&self) -> usize {
        self.frames().len()
    }

    /// Returns the shape of the current frame.
    ///
    /// # Panics
    ///
    /// Panics if the Partial has been poisoned or built, or if there are no frames
    /// (which indicates a bug in the Partial implementation).
    #[inline]
    pub fn shape(&self) -> &'static Shape {
        if self.state != PartialState::Active {
            panic!(
                "Partial::shape() called on non-active Partial (state: {:?})",
                self.state
            );
        }
        self.frames()
            .last()
            .expect("Partial::shape() called but no frames exist - this is a bug")
            .allocated
            .shape()
    }

    /// Returns the shape of the current frame, or `None` if the Partial is
    /// inactive (poisoned or built) or has no frames.
    ///
    /// This is useful for debugging/logging where you want to inspect the state
    /// without risking a panic.
    #[inline]
    pub fn try_shape(&self) -> Option<&'static Shape> {
        if self.state != PartialState::Active {
            return None;
        }
        self.frames().last().map(|f| f.allocated.shape())
    }

    /// Returns the TypePlanCore for this Partial.
    ///
    /// This provides access to the arena-based type plan data, useful for
    /// resolving field lookups and accessing precomputed metadata.
    #[inline]
    pub fn type_plan_core(&self) -> &crate::typeplan::TypePlanCore {
        &self.root_plan
    }

    /// Returns the precomputed StructPlan for the current frame, if available.
    ///
    /// This provides O(1) or O(log n) field lookup instead of O(n) linear scanning.
    /// Returns `None` if:
    /// - The Partial is not active
    /// - The current frame has no TypePlan (e.g., custom deserialization frames)
    /// - The current type is not a struct
    #[inline]
    pub fn struct_plan(&self) -> Option<&crate::typeplan::StructPlan> {
        if self.state != PartialState::Active {
            return None;
        }
        let frame = self.frames().last()?;
        self.root_plan.struct_plan_by_id(frame.type_plan)
    }

    /// Returns the precomputed EnumPlan for the current frame, if available.
    ///
    /// This provides O(1) or O(log n) variant lookup instead of O(n) linear scanning.
    /// Returns `None` if:
    /// - The Partial is not active
    /// - The current type is not an enum
    #[inline]
    pub fn enum_plan(&self) -> Option<&crate::typeplan::EnumPlan> {
        if self.state != PartialState::Active {
            return None;
        }
        let frame = self.frames().last()?;
        self.root_plan.enum_plan_by_id(frame.type_plan)
    }

    /// Returns the precomputed field plans for the current frame.
    ///
    /// This provides access to precomputed validators and default handling without
    /// runtime attribute scanning.
    ///
    /// Returns `None` if the current type is not a struct or enum variant.
    #[inline]
    pub fn field_plans(&self) -> Option<&[crate::typeplan::FieldPlan]> {
        use crate::typeplan::TypePlanNodeKind;
        let frame = self.frames().last().unwrap();
        let node = self.root_plan.node(frame.type_plan);
        match &node.kind {
            TypePlanNodeKind::Struct(struct_plan) => {
                Some(self.root_plan.fields(struct_plan.fields))
            }
            TypePlanNodeKind::Enum(enum_plan) => {
                // For enums, we need the variant index from the tracker
                if let crate::partial::Tracker::Enum { variant_idx, .. } = &frame.tracker {
                    self.root_plan
                        .variants(enum_plan.variants)
                        .get(*variant_idx)
                        .map(|v| self.root_plan.fields(v.fields))
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    /// Returns the precomputed TypePlanNode for the current frame.
    ///
    /// This provides access to the precomputed deserialization strategy and
    /// other metadata computed at Partial allocation time.
    ///
    /// Returns `None` if:
    /// - The Partial is not active
    /// - There are no frames
    #[inline]
    pub fn plan_node(&self) -> Option<&crate::typeplan::TypePlanNode> {
        if self.state != PartialState::Active {
            return None;
        }
        let frame = self.frames().last()?;
        Some(self.root_plan.node(frame.type_plan))
    }

    /// Returns the node ID for the current frame's type plan.
    ///
    /// Returns `None` if:
    /// - The Partial is not active
    /// - There are no frames
    #[inline]
    pub fn plan_node_id(&self) -> Option<crate::typeplan::NodeId> {
        if self.state != PartialState::Active {
            return None;
        }
        let frame = self.frames().last()?;
        Some(frame.type_plan)
    }

    /// Returns the precomputed deserialization strategy for the current frame.
    ///
    /// This tells facet-format exactly how to deserialize the current type without
    /// runtime inspection of Shape/Def/vtable. The strategy is computed once at
    /// TypePlan build time.
    ///
    /// If the current node is a BackRef (recursive type), this automatically
    /// follows the reference to return the target node's strategy.
    ///
    /// Returns `None` if:
    /// - The Partial is not active
    /// - There are no frames
    #[inline]
    pub fn deser_strategy(&self) -> Option<&DeserStrategy> {
        let node = self.plan_node()?;
        // Resolve BackRef if needed - resolve_backref returns the node unchanged if not a BackRef
        let resolved = self.root_plan.resolve_backref(node);
        Some(&resolved.strategy)
    }

    /// Returns the precomputed proxy nodes for the current frame's type.
    ///
    /// These contain TypePlan nodes for all proxies (format-agnostic and format-specific)
    /// on this type, allowing runtime lookup based on format namespace.
    #[inline]
    pub fn proxy_nodes(&self) -> Option<&crate::typeplan::ProxyNodes> {
        let node = self.plan_node()?;
        let resolved = self.root_plan.resolve_backref(node);
        Some(&resolved.proxies)
    }

    /// Returns true if the current frame is building a smart pointer slice (Arc<\[T\]>, Rc<\[T\]>, Box<\[T\]>).
    ///
    /// This is used by deserializers to determine if they should deserialize as a list
    /// rather than recursing into the smart pointer type.
    #[inline]
    pub fn is_building_smart_ptr_slice(&self) -> bool {
        if self.state != PartialState::Active {
            return false;
        }
        self.frames()
            .last()
            .is_some_and(|f| matches!(f.tracker, Tracker::SmartPointerSlice { .. }))
    }

    /// Returns the current path in deferred mode (for debugging/tracing).
    #[inline]
    pub fn current_path(&self) -> Option<facet_path::Path> {
        if self.is_deferred() {
            Some(self.derive_path())
        } else {
            None
        }
    }

    /// Checks if the current frame should be stored for deferred processing.
    ///
    /// This determines whether a frame can safely be stored and re-entered later
    /// in deferred mode. A frame should be stored if:
    /// 1. It's a re-entrant type (struct, enum, collection, Option)
    /// 2. It has storable ownership (Field or Owned)
    /// 3. It doesn't have a SmartPointer parent that needs immediate completion
    ///
    /// Returns `true` if the frame should be stored, `false` if it should be
    /// validated immediately.
    fn should_store_frame_for_deferred(&self) -> bool {
        // In deferred mode, all frames have stable memory and can be stored.
        // PR #2019 added stable storage for all container elements (ListRope for Vec,
        // pending_entries for Map, pending_inner for Option).
        true
    }

    /// Enables deferred materialization mode with the given Resolution.
    ///
    /// When deferred mode is enabled:
    /// - `end()` stores frames instead of validating them
    /// - Re-entering a path restores the stored frame with its state intact
    /// - `finish_deferred()` performs final validation and materialization
    ///
    /// This allows deserializers to handle interleaved fields (e.g., TOML dotted
    /// keys, flattened structs) where nested fields aren't contiguous in the input.
    ///
    /// # Use Cases
    ///
    /// - TOML dotted keys: `inner.x = 1` followed by `count = 2` then `inner.y = 3`
    /// - Flattened structs where nested fields appear at the parent level
    /// - Any format where field order doesn't match struct nesting
    ///
    /// # Errors
    ///
    /// Returns an error if already in deferred mode.
    #[inline]
    pub fn begin_deferred(mut self) -> Result<Self, ReflectError> {
        // Cannot enable deferred mode if already in deferred mode
        if self.is_deferred() {
            return Err(self.err(ReflectErrorKind::InvariantViolation {
                invariant: "begin_deferred() called but already in deferred mode",
            }));
        }

        // Take the stack out of Strict mode and wrap in Deferred mode
        let FrameMode::Strict { stack } = core::mem::replace(
            &mut self.mode,
            FrameMode::Strict { stack: Vec::new() }, // temporary placeholder
        ) else {
            unreachable!("just checked we're not in deferred mode");
        };

        let start_depth = stack.len();
        self.mode = FrameMode::Deferred {
            stack,
            start_depth,
            stored_frames: BTreeMap::new(),
        };
        Ok(self)
    }

    /// Finishes deferred mode: validates all stored frames and finalizes.
    ///
    /// This method:
    /// 1. Validates that all stored frames are fully initialized
    /// 2. Processes frames from deepest to shallowest, updating parent ISets
    /// 3. Validates the root frame
    ///
    /// # Errors
    ///
    /// Returns an error if any required fields are missing or if the partial is
    /// not in deferred mode.
    pub fn finish_deferred(mut self) -> Result<Self, ReflectError> {
        // Check if we're in deferred mode first, before extracting state
        if !self.is_deferred() {
            return Err(self.err(ReflectErrorKind::InvariantViolation {
                invariant: "finish_deferred() called but deferred mode is not enabled",
            }));
        }

        // Extract deferred state, transitioning back to Strict mode
        let FrameMode::Deferred {
            stack,
            mut stored_frames,
            ..
        } = core::mem::replace(&mut self.mode, FrameMode::Strict { stack: Vec::new() })
        else {
            unreachable!("just checked is_deferred()");
        };

        // Restore the stack to self.mode
        self.mode = FrameMode::Strict { stack };

        // Sort paths by depth (deepest first) so we process children before parents.
        // For equal-depth paths, we need stable ordering for list elements:
        // Index(0) must be processed before Index(1) to maintain insertion order.
        let mut paths: Vec<_> = stored_frames.keys().cloned().collect();
        paths.sort_by(|a, b| {
            // Primary: deeper paths first
            let depth_cmp = b.len().cmp(&a.len());
            if depth_cmp != core::cmp::Ordering::Equal {
                return depth_cmp;
            }
            // Secondary: for same-depth paths, compare step by step
            // This ensures Index(0) comes before Index(1) for the same parent
            for (step_a, step_b) in a.steps.iter().zip(b.steps.iter()) {
                let step_cmp = step_a.cmp(step_b);
                if step_cmp != core::cmp::Ordering::Equal {
                    return step_cmp;
                }
            }
            core::cmp::Ordering::Equal
        });

        trace!(
            "finish_deferred: Processing {} stored frames in order: {:?}",
            paths.len(),
            paths
        );

        // Process each stored frame from deepest to shallowest
        for path in paths {
            let mut frame = stored_frames.remove(&path).unwrap();

            trace!(
                "finish_deferred: Processing frame at {:?}, shape {}, tracker {:?}",
                path,
                frame.allocated.shape(),
                frame.tracker.kind()
            );

            // Special handling for SmartPointerSlice: convert builder to Arc<[T]> before validation
            if let Tracker::SmartPointerSlice { vtable, .. } = &frame.tracker {
                let vtable = *vtable;
                let current_shape = frame.allocated.shape();

                // Convert the builder to Arc<[T]>
                let builder_ptr = unsafe { frame.data.assume_init() };
                let arc_ptr = unsafe { (vtable.convert_fn)(builder_ptr) };

                trace!(
                    "finish_deferred: Converting SmartPointerSlice builder to {}",
                    current_shape
                );

                // Handle different ownership cases
                match frame.ownership {
                    FrameOwnership::Field { field_idx } => {
                        // Arc<[T]> is a field in a struct
                        // Find the parent frame and write the Arc to the field location
                        let parent_path = facet_path::Path {
                            shape: path.shape,
                            steps: path.steps[..path.steps.len() - 1].to_vec(),
                        };

                        // Paths are absolute from the root, so the parent frame lives at
                        // stack[parent_path.steps.len()] when it's still on the stack.
                        let parent_frame_opt =
                            if let Some(parent_frame) = stored_frames.get_mut(&parent_path) {
                                Some(parent_frame)
                            } else {
                                self.frames_mut().get_mut(parent_path.steps.len())
                            };

                        if let Some(parent_frame) = parent_frame_opt {
                            // Get the field to find its offset
                            if let Type::User(UserType::Struct(struct_type)) =
                                parent_frame.allocated.shape().ty
                            {
                                let field = &struct_type.fields[field_idx];

                                // Calculate where the Arc should be written (parent.data + field.offset)
                                let field_location =
                                    unsafe { parent_frame.data.field_uninit(field.offset) };

                                // Write the Arc to the parent struct's field location
                                if let Ok(arc_layout) = current_shape.layout.sized_layout() {
                                    let arc_size = arc_layout.size();
                                    unsafe {
                                        core::ptr::copy_nonoverlapping(
                                            arc_ptr.as_byte_ptr(),
                                            field_location.as_mut_byte_ptr(),
                                            arc_size,
                                        );
                                    }

                                    // Free the staging allocation from convert_fn
                                    unsafe {
                                        ::alloc::alloc::dealloc(
                                            arc_ptr.as_byte_ptr() as *mut u8,
                                            arc_layout,
                                        );
                                    }

                                    // Update the frame to point to the correct field location and mark as initialized
                                    frame.data = field_location;
                                    frame.tracker = Tracker::Scalar;
                                    frame.is_init = true;

                                    trace!(
                                        "finish_deferred: SmartPointerSlice converted and written to field {}",
                                        field_idx
                                    );
                                }
                            }
                        }
                    }
                    FrameOwnership::Owned => {
                        // Arc<[T]> is the root - write in place
                        if let Ok(arc_layout) = current_shape.layout.sized_layout() {
                            let arc_size = arc_layout.size();
                            // Allocate new memory for the Arc
                            let new_ptr = facet_core::alloc_for_layout(arc_layout);
                            unsafe {
                                core::ptr::copy_nonoverlapping(
                                    arc_ptr.as_byte_ptr(),
                                    new_ptr.as_mut_byte_ptr(),
                                    arc_size,
                                );
                            }
                            // Free the staging allocation
                            unsafe {
                                ::alloc::alloc::dealloc(
                                    arc_ptr.as_byte_ptr() as *mut u8,
                                    arc_layout,
                                );
                            }
                            frame.data = new_ptr;
                            frame.tracker = Tracker::Scalar;
                            frame.is_init = true;
                        }
                    }
                    _ => {}
                }
            }

            // Fill in defaults for unset fields that have defaults
            if let Err(e) = frame.fill_defaults() {
                // Before cleanup, clear the parent's iset bit for the frame that failed.
                // This prevents the parent from trying to drop this field when Partial is dropped.
                Self::clear_parent_iset_for_path(&path, self.frames_mut(), &mut stored_frames);
                // Consume-time invariant: pending_entries/pending_inner are only populated
                // by the walk after validation succeeds. A frame that fails here hasn't
                // been transferred anywhere, so its own deinit/dealloc is the sole cleanup.
                frame.deinit();
                frame.dealloc();
                // Clean up remaining stored frames safely (deepest first, clearing parent isets)
                Self::cleanup_stored_frames_on_error(stored_frames, self.frames_mut());
                return Err(self.err(e));
            }

            // Validate the frame is fully initialized
            if let Err(e) = frame.require_full_initialization() {
                // Before cleanup, clear the parent's iset bit for the frame that failed.
                // This prevents the parent from trying to drop this field when Partial is dropped.
                Self::clear_parent_iset_for_path(&path, self.frames_mut(), &mut stored_frames);
                // Consume-time invariant: frame hasn't been transferred yet.
                frame.deinit();
                frame.dealloc();
                // Clean up remaining stored frames safely (deepest first, clearing parent isets)
                Self::cleanup_stored_frames_on_error(stored_frames, self.frames_mut());
                return Err(self.err(e));
            }

            // Update parent's ISet to mark this field as initialized.
            // The parent lives either in stored_frames (if it was ended during deferred mode)
            // or on the frames stack at index parent_path.steps.len() (paths are absolute).
            if let Some(last_step) = path.steps.last() {
                // Construct parent path (same shape, all steps except the last one)
                let parent_path = facet_path::Path {
                    shape: path.shape,
                    steps: path.steps[..path.steps.len() - 1].to_vec(),
                };

                // Special handling for Option inner values: when path ends with OptionSome,
                // the parent is an Option frame and we need to complete the Option by
                // writing the inner value into the Option's memory.
                if matches!(last_step, PathStep::OptionSome) {
                    // Find the Option frame (parent)
                    let option_frame =
                        if let Some(parent_frame) = stored_frames.get_mut(&parent_path) {
                            Some(parent_frame)
                        } else {
                            self.frames_mut().get_mut(parent_path.steps.len())
                        };

                    if let Some(option_frame) = option_frame {
                        // The frame contains the inner value - write it into the Option's memory
                        Self::complete_option_frame(option_frame, frame);
                        // Frame data has been transferred to Option - don't drop it
                        continue;
                    }
                }

                // Special handling for SmartPointer inner values: when path ends with Deref,
                // the parent is a SmartPointer frame and we need to complete it by
                // creating the SmartPointer from the inner value.
                if matches!(last_step, PathStep::Deref) {
                    // Find the SmartPointer frame (parent)
                    let smart_ptr_frame =
                        if let Some(parent_frame) = stored_frames.get_mut(&parent_path) {
                            Some(parent_frame)
                        } else {
                            self.frames_mut().get_mut(parent_path.steps.len())
                        };

                    if let Some(smart_ptr_frame) = smart_ptr_frame {
                        // The frame contains the inner value - create the SmartPointer from it
                        Self::complete_smart_pointer_frame(smart_ptr_frame, frame);
                        // Frame data has been transferred to SmartPointer - don't drop it
                        continue;
                    }
                }

                // Special handling for Inner values: when path ends with Inner,
                // the parent is a transparent wrapper (NonZero, ByteString, etc.) and we need
                // to convert the inner value to the parent type using try_from.
                if matches!(last_step, PathStep::Inner) {
                    // Find the parent frame (Inner wrapper)
                    let parent_frame =
                        if let Some(parent_frame) = stored_frames.get_mut(&parent_path) {
                            Some(parent_frame)
                        } else {
                            self.frames_mut().get_mut(parent_path.steps.len())
                        };

                    if let Some(inner_wrapper_frame) = parent_frame {
                        // The frame contains the inner value - convert to parent type using try_from
                        Self::complete_inner_frame(inner_wrapper_frame, frame);
                        // Frame data has been transferred - don't drop it
                        continue;
                    }
                }

                // Special handling for Proxy values: when path ends with Proxy,
                // the parent is the target type (e.g., Inner) and we need to convert
                // the proxy value (e.g., InnerProxy) using the proxy's convert_in.
                if matches!(last_step, PathStep::Proxy) {
                    // Find the parent frame (the proxy target)
                    let parent_frame =
                        if let Some(parent_frame) = stored_frames.get_mut(&parent_path) {
                            Some(parent_frame)
                        } else {
                            self.frames_mut().get_mut(parent_path.steps.len())
                        };

                    if let Some(target_frame) = parent_frame {
                        Self::complete_proxy_frame(target_frame, frame);
                        continue;
                    }
                }

                // Special handling for List/SmartPointerSlice element values: when path
                // ends with Index, the parent is a List or SmartPointerSlice frame and we
                // need to push the element into it. RopeSlot frames live in the parent
                // rope's slot: they don't get "pushed" here (the slot is pre-allocated),
                // but we DO need to mark the slot initialized now that the element has
                // passed validation, so the rope knows it's safe to drop on error.
                if matches!(last_step, PathStep::Index(_)) {
                    // Find the parent frame (List or SmartPointerSlice)
                    let parent_frame =
                        if let Some(parent_frame) = stored_frames.get_mut(&parent_path) {
                            Some(parent_frame)
                        } else {
                            self.frames_mut().get_mut(parent_path.steps.len())
                        };

                    if let Some(parent_frame) = parent_frame {
                        if matches!(frame.ownership, FrameOwnership::RopeSlot) {
                            // Element already lives in rope slot. Mark it initialized now
                            // that validation passed (consume-time). Frame is dropped
                            // silently — no Drop impl, rope owns the buffer.
                            if let Tracker::List {
                                rope: Some(rope), ..
                            } = &mut parent_frame.tracker
                            {
                                rope.mark_last_initialized();
                            }
                            continue;
                        }
                        // Check if parent is a SmartPointerSlice (e.g., Arc<[T]>)
                        if matches!(parent_frame.tracker, Tracker::SmartPointerSlice { .. }) {
                            Self::complete_smart_pointer_slice_item_frame(parent_frame, frame);
                            // Frame data has been transferred to slice builder - don't drop it
                            continue;
                        }
                        // Otherwise try List handling
                        Self::complete_list_item_frame(parent_frame, frame);
                        // Frame data has been transferred to List - don't drop it
                        continue;
                    }
                }

                // Special handling for Map key values: when path ends with MapKey,
                // the parent is a Map frame and we need to push the key into
                // pending_entries at the matching entry_idx.
                if let PathStep::MapKey(entry_idx) = last_step {
                    let entry_idx = *entry_idx;
                    // Find the Map frame (parent)
                    let map_frame = if let Some(parent_frame) = stored_frames.get_mut(&parent_path)
                    {
                        Some(parent_frame)
                    } else {
                        self.frames_mut().get_mut(parent_path.steps.len())
                    };

                    if let Some(map_frame) = map_frame {
                        Self::complete_map_key_frame(map_frame, entry_idx, frame);
                        continue;
                    }
                }

                // Special handling for Map value values: when path ends with MapValue,
                // the parent is a Map frame and we need to fill in the value for the
                // half-entry at the matching entry_idx.
                if let PathStep::MapValue(entry_idx) = last_step {
                    let entry_idx = *entry_idx;
                    // Find the Map frame (parent)
                    let map_frame = if let Some(parent_frame) = stored_frames.get_mut(&parent_path)
                    {
                        Some(parent_frame)
                    } else {
                        self.frames_mut().get_mut(parent_path.steps.len())
                    };

                    if let Some(map_frame) = map_frame {
                        Self::complete_map_value_frame(map_frame, entry_idx, frame);
                        continue;
                    }
                }

                // Only mark field initialized if the step is actually a Field
                if let PathStep::Field(field_idx) = last_step {
                    let field_idx = *field_idx as usize;
                    // Paths are absolute from the root, so the parent frame lives at
                    // stack[parent_path.steps.len()] when it's still on the stack.
                    let parent_frame =
                        if let Some(parent_frame) = stored_frames.get_mut(&parent_path) {
                            Some(parent_frame)
                        } else {
                            self.frames_mut().get_mut(parent_path.steps.len())
                        };
                    if let Some(parent_frame) = parent_frame {
                        Self::mark_field_initialized_by_index(parent_frame, field_idx);
                    }
                }
            }

            // Frame is validated and parent is updated - dealloc if needed
            frame.dealloc();
        }

        // Invariant check: we must have at least one frame after finish_deferred
        if self.frames().is_empty() {
            // No need to poison - returning Err consumes self, Drop will handle cleanup
            return Err(self.err(ReflectErrorKind::InvariantViolation {
                invariant: "finish_deferred() left Partial with no frames",
            }));
        }

        // Fill defaults and validate the root frame is fully initialized
        if let Some(frame) = self.frames_mut().last_mut() {
            // Fill defaults - this can fail if a field has #[facet(default)] but no default impl
            if let Err(e) = frame.fill_defaults() {
                return Err(self.err(e));
            }
            // Root validation failed. At this point, all stored frames have been
            // processed and their parent isets updated.
            // No need to poison - returning Err consumes self, Drop will handle cleanup
            if let Err(e) = frame.require_full_initialization() {
                return Err(self.err(e));
            }
        }

        Ok(self)
    }

    /// Mark a field as initialized in a frame's tracker by index
    fn mark_field_initialized_by_index(frame: &mut Frame, idx: usize) {
        crate::trace!(
            "mark_field_initialized_by_index: idx={}, frame shape={}, tracker={:?}",
            idx,
            frame.allocated.shape(),
            frame.tracker.kind()
        );

        // If the tracker is Scalar but this is a struct type, upgrade to Struct tracker.
        // This can happen if the frame was deinit'd (e.g., by a failed set_default)
        // which resets the tracker to Scalar.
        if matches!(frame.tracker, Tracker::Scalar)
            && let Type::User(UserType::Struct(struct_type)) = frame.allocated.shape().ty
        {
            frame.tracker = Tracker::Struct {
                iset: ISet::new(struct_type.fields.len()),
                current_child: None,
            };
        }

        match &mut frame.tracker {
            Tracker::Struct { iset, .. } => {
                crate::trace!("mark_field_initialized_by_index: setting iset for struct");
                iset.set(idx);
            }
            Tracker::Enum { data, .. } => {
                crate::trace!(
                    "mark_field_initialized_by_index: setting data for enum, before={:?}",
                    data
                );
                data.set(idx);
                crate::trace!(
                    "mark_field_initialized_by_index: setting data for enum, after={:?}",
                    data
                );
            }
            Tracker::Array { iset, .. } => {
                crate::trace!("mark_field_initialized_by_index: setting iset for array");
                iset.set(idx);
            }
            _ => {
                crate::trace!(
                    "mark_field_initialized_by_index: no match for tracker {:?}",
                    frame.tracker.kind()
                );
            }
        }
    }

    /// Clear a parent frame's iset bit for a given path.
    /// The parent could be on the stack or in stored_frames.
    fn clear_parent_iset_for_path(
        path: &Path,
        stack: &mut [Frame],
        stored_frames: &mut ::alloc::collections::BTreeMap<Path, Frame>,
    ) {
        let Some(&PathStep::Field(field_idx)) = path.steps.last() else {
            return;
        };
        let field_idx = field_idx as usize;
        let parent_path = Path {
            shape: path.shape,
            steps: path.steps[..path.steps.len() - 1].to_vec(),
        };

        // Paths are absolute from the root; the frame at a given path lives at
        // stack[path.steps.len()], so the parent lives at stack[parent_path.steps.len()].
        let parent_frame = if let Some(parent_frame) = stored_frames.get_mut(&parent_path) {
            Some(parent_frame)
        } else {
            stack.get_mut(parent_path.steps.len())
        };
        if let Some(parent_frame) = parent_frame {
            Self::unset_field_in_tracker(&mut parent_frame.tracker, field_idx);
        }
    }

    // NOTE: `sever_parent_pending_for_path` has been removed. Under the consume-time
    // pending-population invariant, parent pending slots (pending_entries, pending_inner,
    // etc.) are only populated by the walk in `finish_deferred` AFTER a child frame's
    // validation passes. A failing child frame has its buffer still owned by itself, so
    // `frame.deinit(); frame.dealloc()` is the complete cleanup. No parent-side sever
    // is required.

    /// Helper to unset a field index in a tracker's iset
    fn unset_field_in_tracker(tracker: &mut Tracker, field_idx: usize) {
        match tracker {
            Tracker::Struct { iset, .. } => {
                iset.unset(field_idx);
            }
            Tracker::Enum { data, .. } => {
                data.unset(field_idx);
            }
            Tracker::Array { iset, .. } => {
                iset.unset(field_idx);
            }
            _ => {}
        }
    }

    /// Safely clean up stored frames on error in finish_deferred.
    ///
    /// This mirrors the cleanup logic in Drop: process frames deepest-first and
    /// clear parent's iset bits before deiniting children to prevent double-drops.
    fn cleanup_stored_frames_on_error(
        mut stored_frames: ::alloc::collections::BTreeMap<Path, Frame>,
        stack: &mut [Frame],
    ) {
        // Sort by depth (deepest first) so children are processed before parents
        let mut paths: Vec<_> = stored_frames.keys().cloned().collect();
        paths.sort_by_key(|p| core::cmp::Reverse(p.steps.len()));

        trace!(
            "cleanup_stored_frames_on_error: {} frames to clean, paths: {:?}",
            paths.len(),
            paths
        );

        for path in &paths {
            if let Some(frame) = stored_frames.get(path) {
                trace!(
                    "cleanup: processing path={:?}, shape={}, tracker={:?}, is_init={}, ownership={:?}",
                    path,
                    frame.allocated.shape(),
                    frame.tracker.kind(),
                    frame.is_init,
                    frame.ownership,
                );
                // Dump iset contents for struct/enum trackers
                match &frame.tracker {
                    Tracker::Struct { iset: _iset, .. } => {
                        trace!("cleanup:   Struct iset = {:?}", _iset);
                    }
                    Tracker::Enum {
                        variant: _variant,
                        data: _data,
                        ..
                    } => {
                        trace!("cleanup:   Enum {:?} data = {:?}", _variant.name, _data);
                    }
                    _ => {}
                }
            }
        }

        for path in paths {
            if let Some(mut frame) = stored_frames.remove(&path) {
                trace!(
                    "cleanup: REMOVING path={:?}, shape={}, tracker={:?}",
                    path,
                    frame.allocated.shape(),
                    frame.tracker.kind(),
                );
                // Before dropping this frame, clear the parent's iset bit so the
                // parent won't try to drop this field again.
                Self::clear_parent_iset_for_path(&path, stack, &mut stored_frames);
                // Under the consume-time SSoT invariant, stored frames always own
                // their own buffer (parent pending slots are only populated at walk
                // consume-time, after validation). Standard deinit + dealloc handles
                // cleanup; no parent pending-slot severing is needed.
                trace!("cleanup: calling deinit() on path={:?}", path,);
                frame.deinit();
                frame.dealloc();
            }
        }
    }

    /// Complete an Option frame by writing the inner value and marking it initialized.
    /// Used in finish_deferred when processing a stored frame at a path ending with "Some".
    fn complete_option_frame(option_frame: &mut Frame, inner_frame: Frame) {
        if let Def::Option(option_def) = option_frame.allocated.shape().def {
            // Use the Option vtable to initialize Some(inner_value)
            let init_some_fn = option_def.vtable.init_some;

            // The inner frame contains the inner value
            let inner_value_ptr = unsafe { inner_frame.data.assume_init() };

            // Initialize the Option as Some(inner_value)
            unsafe {
                init_some_fn(option_frame.data, inner_value_ptr);
            }

            // Deallocate the inner value's memory since init_some_fn moved it
            if let FrameOwnership::Owned = inner_frame.ownership
                && let Ok(layout) = inner_frame.allocated.shape().layout.sized_layout()
                && layout.size() > 0
            {
                unsafe {
                    ::alloc::alloc::dealloc(inner_frame.data.as_mut_byte_ptr(), layout);
                }
            }

            // Mark the Option as initialized
            option_frame.tracker = Tracker::Option {
                building_inner: false,
                pending_inner: None,
            };
            option_frame.is_init = true;
        }
    }

    fn complete_smart_pointer_frame(smart_ptr_frame: &mut Frame, inner_frame: Frame) {
        if let Def::Pointer(smart_ptr_def) = smart_ptr_frame.allocated.shape().def {
            // Use the SmartPointer vtable to finalize the smart pointer from the inner value.
            if let Some(new_into_fn) = smart_ptr_def.vtable.new_into_fn {
                // Sized pointee case: use new_into_fn
                let _ = unsafe { inner_frame.data.assume_init() };

                // Create the SmartPointer with the inner value
                unsafe {
                    new_into_fn(
                        smart_ptr_frame.data,
                        PtrMut::new(inner_frame.data.as_mut_byte_ptr()),
                    );
                }

                // Deallocate the inner value's memory since new_into_fn moved it
                if let FrameOwnership::Owned = inner_frame.ownership
                    && let Ok(layout) = inner_frame.allocated.shape().layout.sized_layout()
                    && layout.size() > 0
                {
                    unsafe {
                        ::alloc::alloc::dealloc(inner_frame.data.as_mut_byte_ptr(), layout);
                    }
                }

                // Mark the SmartPointer as initialized
                smart_ptr_frame.tracker = Tracker::SmartPointer {
                    building_inner: false,
                    pending_inner: None,
                };
                smart_ptr_frame.is_init = true;
            } else if Frame::try_borrow_and_promote_smart_pointer(
                smart_ptr_def,
                smart_ptr_frame.data,
                inner_frame.allocated.shape(),
                inner_frame.data,
            ) {
                // Promotion has initialized the independent destination. Record that
                // before dropping the source, whose destructor may run user code.
                smart_ptr_frame.tracker = Tracker::SmartPointer {
                    building_inner: false,
                    pending_inner: None,
                };
                smart_ptr_frame.is_init = true;

                // The source was validated as fully initialized before this call.
                // Drop the whole value, including an enum's custom destructor, rather
                // than using partial-initialization cleanup for its individual fields.
                PendingSmartPointerInner::from_initialized_frame(inner_frame).drop_and_dealloc();
            } else if let Some(pointee) = smart_ptr_def.pointee()
                && pointee.is_shape(str::SHAPE)
                && inner_frame.allocated.shape().is_shape(String::SHAPE)
            {
                // Unsized pointee case: String -> Arc<str>/Box<str>/Rc<str> conversion
                use ::alloc::{borrow::Cow, rc::Rc, string::String, sync::Arc};
                use facet_core::KnownPointer;

                let Some(known) = smart_ptr_def.known else {
                    return;
                };

                // Read the String value from the inner frame
                let string_ptr = inner_frame.data.as_mut_byte_ptr() as *mut String;
                let string_value = unsafe { core::ptr::read(string_ptr) };

                // Convert to the appropriate smart pointer type
                match known {
                    KnownPointer::Box => {
                        let boxed: ::alloc::boxed::Box<str> = string_value.into_boxed_str();
                        unsafe {
                            core::ptr::write(
                                smart_ptr_frame.data.as_mut_byte_ptr()
                                    as *mut ::alloc::boxed::Box<str>,
                                boxed,
                            );
                        }
                    }
                    KnownPointer::Arc => {
                        let arc: Arc<str> = Arc::from(string_value.into_boxed_str());
                        unsafe {
                            core::ptr::write(
                                smart_ptr_frame.data.as_mut_byte_ptr() as *mut Arc<str>,
                                arc,
                            );
                        }
                    }
                    KnownPointer::Rc => {
                        let rc: Rc<str> = Rc::from(string_value.into_boxed_str());
                        unsafe {
                            core::ptr::write(
                                smart_ptr_frame.data.as_mut_byte_ptr() as *mut Rc<str>,
                                rc,
                            );
                        }
                    }
                    KnownPointer::Cow => {
                        let cow: Cow<'static, str> = Cow::Owned(string_value);
                        unsafe {
                            core::ptr::write(
                                smart_ptr_frame.data.as_mut_byte_ptr() as *mut Cow<'static, str>,
                                cow,
                            );
                        }
                    }
                    _ => return,
                }

                // Deallocate the String's memory (we moved the data out via ptr::read)
                if let FrameOwnership::Owned = inner_frame.ownership
                    && let Ok(layout) = inner_frame.allocated.shape().layout.sized_layout()
                    && layout.size() > 0
                {
                    unsafe {
                        ::alloc::alloc::dealloc(inner_frame.data.as_mut_byte_ptr(), layout);
                    }
                }

                // Mark the SmartPointer as initialized
                smart_ptr_frame.tracker = Tracker::SmartPointer {
                    building_inner: false,
                    pending_inner: None,
                };
                smart_ptr_frame.is_init = true;
            }
        }
    }

    /// Complete an Inner frame by converting the inner value to the parent type using try_from
    /// (for deferred finalization)
    fn complete_inner_frame(inner_wrapper_frame: &mut Frame, inner_frame: Frame) {
        let wrapper_shape = inner_wrapper_frame.allocated.shape();
        let inner_ptr = PtrConst::new(inner_frame.data.as_byte_ptr());
        let inner_shape = inner_frame.allocated.shape();

        // Handle Direct and Indirect vtables - both return TryFromOutcome
        let result = match wrapper_shape.vtable {
            facet_core::VTableErased::Direct(vt) => {
                if let Some(try_from_fn) = vt.try_from {
                    unsafe {
                        try_from_fn(
                            inner_wrapper_frame.data.as_mut_byte_ptr() as *mut (),
                            inner_shape,
                            inner_ptr,
                        )
                    }
                } else {
                    return;
                }
            }
            facet_core::VTableErased::Indirect(vt) => {
                if let Some(try_from_fn) = vt.try_from {
                    let ox_uninit =
                        facet_core::OxPtrUninit::new(inner_wrapper_frame.data, wrapper_shape);
                    unsafe { try_from_fn(ox_uninit, inner_shape, inner_ptr) }
                } else {
                    return;
                }
            }
            // Unknown vtable kind: no conversion possible.
            _ => return,
        };

        match result {
            TryFromOutcome::Converted => {
                crate::trace!(
                    "complete_inner_frame: converted {} to {}",
                    inner_shape,
                    wrapper_shape
                );
            }
            // Treat unsupported, failed, and unknown outcomes as not-converted.
            _ => {
                crate::trace!(
                    "complete_inner_frame: conversion failed from {} to {}",
                    inner_shape,
                    wrapper_shape
                );
                return;
            }
        }

        // Deallocate the inner value's memory (try_from consumed it)
        if let FrameOwnership::Owned = inner_frame.ownership
            && let Ok(layout) = inner_frame.allocated.shape().layout.sized_layout()
            && layout.size() > 0
        {
            unsafe {
                ::alloc::alloc::dealloc(inner_frame.data.as_mut_byte_ptr(), layout);
            }
        }

        // Mark the wrapper as initialized
        inner_wrapper_frame.tracker = Tracker::Scalar;
        inner_wrapper_frame.is_init = true;
    }

    /// Complete a proxy conversion during deferred finalization.
    ///
    /// This handles proxy types (e.g., `#[facet(proxy = InnerProxy)]`) that were
    /// deferred during flatten deserialization. The proxy frame's children (e.g.,
    /// `Vec<f64>` fields) have already been materialized (ropes drained), so it's
    /// now safe to run the conversion.
    fn complete_proxy_frame(target_frame: &mut Frame, proxy_frame: Frame) {
        // Get the convert_in function from the proxy stored on the frame
        let Some(proxy_def) = proxy_frame.shape_level_proxy else {
            crate::trace!(
                "complete_proxy_frame: no shape_level_proxy on frame {}",
                proxy_frame.allocated.shape()
            );
            return;
        };
        let convert_in = proxy_def.convert_in;

        let _proxy_shape = proxy_frame.allocated.shape();
        let _target_shape = target_frame.allocated.shape();

        crate::trace!(
            "complete_proxy_frame: converting {} to {}",
            _proxy_shape,
            _target_shape
        );

        unsafe {
            let inner_value_ptr = proxy_frame.data.assume_init().as_const();
            let res = (convert_in)(inner_value_ptr, target_frame.data);

            match res {
                Ok(rptr) => {
                    if rptr.as_uninit() != target_frame.data {
                        crate::trace!(
                            "complete_proxy_frame: convert_in returned unexpected pointer"
                        );
                        return;
                    }
                }
                Err(_message) => {
                    crate::trace!("complete_proxy_frame: conversion failed: {}", _message);
                    return;
                }
            }
        }

        // Deallocate the proxy frame's memory (convert_in consumed it via ptr::read)
        if let FrameOwnership::Owned = proxy_frame.ownership
            && let Ok(layout) = proxy_frame.allocated.shape().layout.sized_layout()
            && layout.size() > 0
        {
            unsafe {
                ::alloc::alloc::dealloc(proxy_frame.data.as_mut_byte_ptr(), layout);
            }
        }

        // Mark the target as initialized
        target_frame.is_init = true;
    }

    /// Complete a List frame by pushing an element into it (for deferred finalization)
    fn complete_list_item_frame(list_frame: &mut Frame, element_frame: Frame) {
        if let Def::List(list_def) = list_frame.allocated.shape().def
            && let Some(push_fn) = list_def.push()
        {
            // The element frame contains the element value
            let element_ptr = PtrMut::new(element_frame.data.as_mut_byte_ptr());

            // Use push to add element to the list
            unsafe {
                push_fn(PtrMut::new(list_frame.data.as_mut_byte_ptr()), element_ptr);
            }

            crate::trace!(
                "complete_list_item_frame: pushed element into {}",
                list_frame.allocated.shape()
            );

            // Deallocate the element's memory since push moved it
            if let FrameOwnership::Owned = element_frame.ownership
                && let Ok(layout) = element_frame.allocated.shape().layout.sized_layout()
                && layout.size() > 0
            {
                unsafe {
                    ::alloc::alloc::dealloc(element_frame.data.as_mut_byte_ptr(), layout);
                }
            }
        }
    }

    /// Complete a SmartPointerSlice element frame by pushing the element into the slice builder
    /// (for deferred finalization)
    fn complete_smart_pointer_slice_item_frame(
        slice_frame: &mut Frame,
        element_frame: Frame,
    ) -> bool {
        if let Tracker::SmartPointerSlice { vtable, .. } = &slice_frame.tracker {
            let vtable = *vtable;
            // The slice frame's data pointer IS the builder pointer
            let builder_ptr = slice_frame.data;

            // Push the element into the builder
            unsafe {
                (vtable.push_fn)(
                    PtrMut::new(builder_ptr.as_mut_byte_ptr()),
                    PtrMut::new(element_frame.data.as_mut_byte_ptr()),
                );
            }

            crate::trace!(
                "complete_smart_pointer_slice_item_frame: pushed element into builder for {}",
                slice_frame.allocated.shape()
            );

            // Deallocate the element's memory since push moved it
            if let FrameOwnership::Owned = element_frame.ownership
                && let Ok(layout) = element_frame.allocated.shape().layout.sized_layout()
                && layout.size() > 0
            {
                unsafe {
                    ::alloc::alloc::dealloc(element_frame.data.as_mut_byte_ptr(), layout);
                }
            }
            return true;
        }
        false
    }

    /// Complete a Map key frame by transferring the key buffer into `pending_entries`
    /// at position `entry_idx` as a half-entry `(key_ptr, None)`
    /// (for deferred finalization, walk consume-time).
    ///
    /// Called only from the `finish_deferred` walk, after `require_full_initialization`
    /// has validated the key frame. The `entry_idx` comes from the `PathStep::MapKey(idx)`
    /// and matches the map's `current_entry_index` assigned at `begin_key` time.
    /// MapKey frames for the same map are visited in ascending idx order, so pushing
    /// is equivalent to indexed insertion (asserted below).
    ///
    /// `key_frame` is dropped silently on return — Frame has no Drop impl, so
    /// `pending_entries` becomes the sole owner of this buffer.
    fn complete_map_key_frame(map_frame: &mut Frame, entry_idx: u32, key_frame: Frame) {
        if let Tracker::Map {
            pending_entries, ..
        } = &mut map_frame.tracker
        {
            debug_assert_eq!(
                pending_entries.len(),
                entry_idx as usize,
                "MapKey frames must arrive in ascending entry_idx order"
            );
            pending_entries.push((key_frame.data, None));
            crate::trace!(
                "complete_map_key_frame: pushed half-entry at idx {} for {}",
                entry_idx,
                map_frame.allocated.shape()
            );
        }
    }

    /// Complete a Map value frame by upgrading the half-entry at position `entry_idx`
    /// `(key_ptr, None)` to a full `(key_ptr, Some(value_ptr))`
    /// (for deferred finalization, walk consume-time).
    ///
    /// Called only from the `finish_deferred` walk, after `require_full_initialization`
    /// has validated the value frame. The `entry_idx` comes from
    /// `PathStep::MapValue(idx)` and indexes the matching half-entry placed by
    /// `complete_map_key_frame`. `value_frame` is dropped silently on return —
    /// Frame has no Drop impl, so `pending_entries` becomes the sole owner of the
    /// buffer.
    fn complete_map_value_frame(map_frame: &mut Frame, entry_idx: u32, value_frame: Frame) {
        if let Tracker::Map {
            pending_entries, ..
        } = &mut map_frame.tracker
        {
            let slot = pending_entries
                .get_mut(entry_idx as usize)
                .expect("pending_entries must have a half-entry at entry_idx");
            debug_assert!(
                slot.1.is_none(),
                "pending entry at entry_idx must be a half-entry (None value), got Some"
            );
            slot.1 = Some(value_frame.data);
            crate::trace!(
                "complete_map_value_frame: upgraded half-entry at idx {} to full entry for {}",
                entry_idx,
                map_frame.allocated.shape()
            );
        }
    }

    /// Pops the current frame off the stack, indicating we're done initializing the current field
    pub fn end(mut self) -> Result<Self, ReflectError> {
        // FAST PATH: Handle the common case of ending a simple scalar field in a struct.
        // This avoids all the edge-case checks (SmartPointerSlice, deferred mode, custom
        // deserialization, etc.) that dominate the slow path.
        if self.frames().len() >= 2 && !self.is_deferred() {
            let frames = self.frames_mut();
            let top_idx = frames.len() - 1;
            let parent_idx = top_idx - 1;

            // Check if this is a simple scalar field being returned to a struct parent
            if let (
                Tracker::Scalar,
                true, // is_init
                FrameOwnership::Field { field_idx },
                false, // not using custom deserialization
            ) = (
                &frames[top_idx].tracker,
                frames[top_idx].is_init,
                frames[top_idx].ownership,
                frames[top_idx].using_custom_deserialization,
            ) && let Tracker::Struct {
                iset,
                current_child,
            } = &mut frames[parent_idx].tracker
            {
                // Fast path: just update parent's iset and pop
                iset.set(field_idx);
                *current_child = None;
                frames.pop();
                return Ok(self);
            }
        }

        // SLOW PATH: Handle all the edge cases

        // Strategic tracing: show the frame stack state
        #[cfg(feature = "tracing")]
        {
            use ::alloc::string::ToString;
            let frames = self.frames();
            let stack_desc: Vec<_> = frames
                .iter()
                .map(|f| ::alloc::format!("{}({:?})", f.allocated.shape(), f.tracker.kind()))
                .collect();
            let path = if self.is_deferred() {
                ::alloc::format!("{:?}", self.derive_path())
            } else {
                "N/A".to_string()
            };
            crate::trace!(
                "end() SLOW PATH: stack=[{}], deferred={}, path={}",
                stack_desc.join(" > "),
                self.is_deferred(),
                path
            );
        }

        // Special handling for SmartPointerSlice - convert builder to Arc
        // Check if the current (top) frame is a SmartPointerSlice that needs conversion
        let needs_slice_conversion = {
            let frames = self.frames();
            if frames.is_empty() {
                false
            } else {
                let top_idx = frames.len() - 1;
                matches!(
                    frames[top_idx].tracker,
                    Tracker::SmartPointerSlice {
                        building_item: false,
                        ..
                    }
                )
            }
        };

        if needs_slice_conversion {
            // In deferred mode, don't convert immediately - let finish_deferred handle it.
            // Set building_item = true and return early (matching non-deferred behavior).
            // The next end() call will store the frame.
            if self.is_deferred() {
                let frames = self.frames_mut();
                let top_idx = frames.len() - 1;
                if let Tracker::SmartPointerSlice { building_item, .. } =
                    &mut frames[top_idx].tracker
                {
                    *building_item = true;
                }
                return Ok(self);
            } else {
                // Get shape info upfront to avoid borrow conflicts
                let current_shape = self.frames().last().unwrap().allocated.shape();

                let frames = self.frames_mut();
                let top_idx = frames.len() - 1;

                if let Tracker::SmartPointerSlice { vtable, .. } = &frames[top_idx].tracker {
                    // Convert the builder to Arc<[T]>
                    let vtable = *vtable;
                    let builder_ptr = unsafe { frames[top_idx].data.assume_init() };
                    let arc_ptr = unsafe { (vtable.convert_fn)(builder_ptr) };

                    match frames[top_idx].ownership {
                        FrameOwnership::Field { field_idx } => {
                            // Arc<[T]> is a field in a struct
                            // The field frame's original data pointer was overwritten with the builder pointer,
                            // so we need to reconstruct where the Arc should be written.

                            // Get parent frame and field info
                            let parent_idx = top_idx - 1;
                            let parent_frame = &frames[parent_idx];

                            // Get the field to find its offset
                            let field = if let Type::User(UserType::Struct(struct_type)) =
                                parent_frame.allocated.shape().ty
                            {
                                &struct_type.fields[field_idx]
                            } else {
                                return Err(self.err(ReflectErrorKind::InvariantViolation {
                                invariant: "SmartPointerSlice field frame parent must be a struct",
                            }));
                            };

                            // Calculate where the Arc should be written (parent.data + field.offset)
                            let field_location =
                                unsafe { parent_frame.data.field_uninit(field.offset) };

                            // Write the Arc to the parent struct's field location
                            let arc_layout = match current_shape.layout.sized_layout() {
                                Ok(layout) => layout,
                                Err(_) => {
                                    return Err(self.err(ReflectErrorKind::Unsized {
                                    shape: current_shape,
                                    operation: "SmartPointerSlice conversion requires sized Arc",
                                }));
                                }
                            };
                            let arc_size = arc_layout.size();
                            unsafe {
                                core::ptr::copy_nonoverlapping(
                                    arc_ptr.as_byte_ptr(),
                                    field_location.as_mut_byte_ptr(),
                                    arc_size,
                                );
                            }

                            // Free the staging allocation from convert_fn (the Arc was copied to field_location)
                            unsafe {
                                ::alloc::alloc::dealloc(
                                    arc_ptr.as_byte_ptr() as *mut u8,
                                    arc_layout,
                                );
                            }

                            // Update the frame to point to the correct field location and mark as initialized
                            frames[top_idx].data = field_location;
                            frames[top_idx].tracker = Tracker::Scalar;
                            frames[top_idx].is_init = true;

                            // Return WITHOUT popping - the field frame will be popped by the next end() call
                            return Ok(self);
                        }
                        FrameOwnership::Owned => {
                            // Arc<[T]> is the root type or owned independently
                            // The frame already has the allocation, we just need to update it with the Arc

                            // The frame's data pointer is currently the builder, but we allocated
                            // the Arc memory in the convert_fn. Update to point to the Arc.
                            frames[top_idx].data = PtrUninit::new(arc_ptr.as_byte_ptr() as *mut u8);
                            frames[top_idx].tracker = Tracker::Scalar;
                            frames[top_idx].is_init = true;
                            // Keep Owned ownership so Guard will properly deallocate

                            // Return WITHOUT popping - the frame stays and will be built/dropped normally
                            return Ok(self);
                        }
                        FrameOwnership::TrackedBuffer
                        | FrameOwnership::BorrowedInPlace
                        | FrameOwnership::External
                        | FrameOwnership::RopeSlot => {
                            return Err(self.err(ReflectErrorKind::InvariantViolation {
                            invariant: "SmartPointerSlice cannot have TrackedBuffer/BorrowedInPlace/External/RopeSlot ownership after conversion",
                        }));
                        }
                    }
                }
            }
        }

        if self.frames().len() <= 1 {
            // Never pop the last/root frame - this indicates a broken state machine
            // No need to poison - returning Err consumes self, Drop will handle cleanup
            return Err(self.err(ReflectErrorKind::InvariantViolation {
                invariant: "Partial::end() called with only one frame on the stack",
            }));
        }

        // In deferred mode, cannot pop below the start depth
        if let Some(start_depth) = self.start_depth()
            && self.frames().len() <= start_depth
        {
            // No need to poison - returning Err consumes self, Drop will handle cleanup
            return Err(self.err(ReflectErrorKind::InvariantViolation {
                invariant: "Partial::end() called but would pop below deferred start depth",
            }));
        }

        // Require that the top frame is fully initialized before popping.
        // In deferred mode, tracked frames (those that will be stored for re-entry)
        // defer validation to finish_deferred(). All other frames validate now
        // using the TypePlan's FillRule (which knows what's Required vs Defaultable).
        let requires_full_init = if !self.is_deferred() {
            true
        } else {
            // If this frame will be stored, defer validation to finish_deferred().
            // Otherwise validate now.
            !self.should_store_frame_for_deferred()
        };

        if requires_full_init {
            // Try the optimized path using precomputed FieldInitPlan
            // Extract frame info first (borrows only self.mode)
            let frame_info = self.mode.stack().last().map(|frame| {
                let variant_idx = match &frame.tracker {
                    Tracker::Enum { variant_idx, .. } => Some(*variant_idx),
                    _ => None,
                };
                (frame.type_plan, variant_idx)
            });

            // Look up plans from the type plan node - need to resolve NodeId to get the actual node
            let plans_info = frame_info.and_then(|(type_plan_id, variant_idx)| {
                let type_plan = self.root_plan.node(type_plan_id);
                match &type_plan.kind {
                    TypePlanNodeKind::Struct(struct_plan) => Some(struct_plan.fields),
                    TypePlanNodeKind::Enum(enum_plan) => {
                        let variants = self.root_plan.variants(enum_plan.variants);
                        variant_idx.and_then(|idx| variants.get(idx).map(|v| v.fields))
                    }
                    _ => None,
                }
            });

            if let Some(plans_range) = plans_info {
                // Resolve the SliceRange to an actual slice
                let plans = self.root_plan.fields(plans_range);
                // Now mutably borrow mode.stack to get the frame
                // (root_plan borrow of `plans` is still active but that's fine -
                // mode and root_plan are separate fields)
                let frame = self.mode.stack_mut().last_mut().unwrap();
                frame
                    .fill_and_require_fields(plans, plans.len(), &self.root_plan)
                    .map_err(|e| self.err(e))?;
            } else {
                // Fall back to the old path if optimized path wasn't available
                if let Some(frame) = self.frames_mut().last_mut() {
                    frame.fill_defaults().map_err(|e| self.err(e))?;
                }

                let frame = self.frames_mut().last_mut().unwrap();
                let result = frame.require_full_initialization();
                if result.is_err() {
                    crate::trace!(
                        "end() VALIDATION FAILED: {} ({:?}) is_init={} - {:?}",
                        frame.allocated.shape(),
                        frame.tracker.kind(),
                        frame.is_init,
                        result
                    );
                }
                result.map_err(|e| self.err(e))?
            }
        }

        // In deferred mode, check if we should store this frame for potential re-entry.
        // We need to compute the storage path BEFORE popping so we can check it.
        //
        // Store frames that can be re-entered in deferred mode.
        // This includes structs, enums, collections, and Options (which need to be
        // stored so finish_deferred can find them when processing their inner values).
        let deferred_storage_info = if self.is_deferred() {
            let should_store = self.should_store_frame_for_deferred();

            if should_store {
                // Compute the "field-only" path for storage by finding all Field steps
                // from PARENT frames only. The frame being ended shouldn't contribute to
                // its own path (its current_child points to ITS children, not to itself).
                //
                // Note: We include ALL frames in the path computation (including those
                // before start_depth) because they contain navigation info. The start_depth
                // only determines which frames we STORE, not which frames contribute to paths.
                //
                // Get the root shape for the Path from the first frame
                let root_shape = self
                    .frames()
                    .first()
                    .map(|f| f.allocated.shape())
                    .unwrap_or_else(|| <() as facet_core::Facet>::SHAPE);

                let mut field_path = facet_path::Path::new(root_shape);
                let frames_len = self.frames().len();
                // Iterate over all frames EXCEPT the last one (the one being ended)
                for (frame_idx, frame) in self.frames().iter().enumerate() {
                    // Skip the frame being ended
                    if frame_idx == frames_len - 1 {
                        continue;
                    }
                    // Extract navigation steps from frames
                    // This MUST match derive_path() for consistency
                    match &frame.tracker {
                        Tracker::Struct {
                            current_child: Some(idx),
                            ..
                        } => {
                            field_path.push(PathStep::Field(*idx as u32));
                        }
                        Tracker::Enum {
                            current_child: Some(idx),
                            ..
                        } => {
                            field_path.push(PathStep::Field(*idx as u32));
                        }
                        Tracker::List {
                            current_child: Some(idx),
                            ..
                        } => {
                            field_path.push(PathStep::Index(*idx as u32));
                        }
                        Tracker::Array {
                            current_child: Some(idx),
                            ..
                        } => {
                            field_path.push(PathStep::Index(*idx as u32));
                        }
                        Tracker::Option {
                            building_inner: true,
                            ..
                        } => {
                            // Option with building_inner contributes OptionSome to path
                            field_path.push(PathStep::OptionSome);
                        }
                        Tracker::SmartPointer {
                            building_inner: true,
                            ..
                        } => {
                            // SmartPointer with building_inner contributes Deref to path
                            field_path.push(PathStep::Deref);
                        }
                        Tracker::SmartPointerSlice {
                            current_child: Some(idx),
                            ..
                        } => {
                            // SmartPointerSlice with current_child contributes Index to path
                            field_path.push(PathStep::Index(*idx as u32));
                        }
                        Tracker::Inner {
                            building_inner: true,
                        } => {
                            // Inner with building_inner contributes Inner to path
                            field_path.push(PathStep::Inner);
                        }
                        Tracker::Map {
                            current_entry_index: Some(idx),
                            building_key,
                            ..
                        } => {
                            // Map with active entry contributes MapKey or MapValue with entry index
                            if *building_key {
                                field_path.push(PathStep::MapKey(*idx as u32));
                            } else {
                                field_path.push(PathStep::MapValue(*idx as u32));
                            }
                        }
                        _ => {}
                    }

                    // If the next frame on the stack is a proxy frame, add a Proxy
                    // path step. This distinguishes the proxy frame (and its children)
                    // from the parent frame that the proxy writes into, preventing path
                    // collisions in deferred mode where both frames are stored.
                    if frame_idx + 1 < frames_len
                        && self.frames()[frame_idx + 1].using_custom_deserialization
                    {
                        field_path.push(PathStep::Proxy);
                    }
                }

                if !field_path.is_empty() {
                    Some(field_path)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Pop the frame and save its data pointer for SmartPointer handling
        let mut popped_frame = self.frames_mut().pop().unwrap();

        // In non-deferred mode, proxy frames are processed immediately.
        // In deferred mode, proxy frames are stored (with a PathStep::Proxy
        // distinguishing them from their parent) and the conversion is handled
        // by finish_deferred after children have been fully materialized.
        if popped_frame.using_custom_deserialization && deferred_storage_info.is_none() {
            // First check the proxy stored in the frame (used for format-specific proxies
            // and container-level proxies), then fall back to field-level proxy.
            // This ordering is important because format-specific proxies store their
            // proxy in shape_level_proxy, and we want them to take precedence over
            // the format-agnostic field.proxy().
            let deserialize_with: Option<facet_core::ProxyConvertInFn> =
                popped_frame.shape_level_proxy.map(|p| p.convert_in);

            // Fall back to field-level proxy (format-agnostic)
            let deserialize_with = deserialize_with.or_else(|| {
                self.parent_field()
                    .and_then(|f| f.proxy().map(|p| p.convert_in))
            });

            if let Some(deserialize_with) = deserialize_with {
                // Get parent shape upfront to avoid borrow conflicts
                let parent_shape = self.frames().last().unwrap().allocated.shape();
                let parent_frame = self.frames_mut().last_mut().unwrap();

                trace!(
                    "Detected custom conversion needed from {} to {}",
                    popped_frame.allocated.shape(),
                    parent_shape
                );

                unsafe {
                    let res = {
                        let inner_value_ptr = popped_frame.data.assume_init().as_const();
                        (deserialize_with)(inner_value_ptr, parent_frame.data)
                    };
                    let popped_frame_shape = popped_frame.allocated.shape();

                    // Note: We do NOT call deinit() here because deserialize_with uses
                    // ptr::read to take ownership of the source value. Calling deinit()
                    // would cause a double-free. We mark is_init as false to satisfy
                    // dealloc()'s assertion, then deallocate the memory.
                    popped_frame.is_init = false;
                    popped_frame.dealloc();
                    let parent_data = parent_frame.data;
                    match res {
                        Ok(rptr) => {
                            if rptr.as_uninit() != parent_data {
                                return Err(self.err(
                                    ReflectErrorKind::CustomDeserializationError {
                                        message:
                                            "deserialize_with did not return the expected pointer"
                                                .into(),
                                        src_shape: popped_frame_shape,
                                        dst_shape: parent_shape,
                                    },
                                ));
                            }
                        }
                        Err(message) => {
                            return Err(self.err(ReflectErrorKind::CustomDeserializationError {
                                message,
                                src_shape: popped_frame_shape,
                                dst_shape: parent_shape,
                            }));
                        }
                    }
                    // Re-borrow parent_frame after potential early returns
                    let parent_frame = self.frames_mut().last_mut().unwrap();
                    parent_frame.mark_as_init();
                }
                return Ok(self);
            }
        }

        // If we determined this frame should be stored for deferred re-entry, do it now
        if let Some(storage_path) = deferred_storage_info {
            trace!(
                "end(): Storing frame for deferred path {:?}, shape {}",
                storage_path,
                popped_frame.allocated.shape()
            );

            if let FrameMode::Deferred {
                stack,
                stored_frames,
                ..
            } = &mut self.mode
            {
                // Mark the field as initialized in the parent frame.
                // This is important because the parent might validate before
                // finish_deferred runs (e.g., parent is an array element that
                // isn't stored). Without this, the parent's validation would
                // fail with "missing field".
                if let FrameOwnership::Field { field_idx } = popped_frame.ownership
                    && let Some(parent_frame) = stack.last_mut()
                {
                    Self::mark_field_initialized_by_index(parent_frame, field_idx);
                }

                // For BorrowedInPlace DynamicValue frames (e.g., re-entered pending entries),
                // flush pending_elements/pending_entries and return without storing.
                // These frames point to memory that's already tracked in the parent's
                // pending_entries - storing them would overwrite the entry.
                if matches!(popped_frame.ownership, FrameOwnership::BorrowedInPlace) {
                    crate::trace!(
                        "end(): BorrowedInPlace frame, flushing pending items and returning"
                    );
                    if let Err(kind) = popped_frame.require_full_initialization() {
                        return Err(ReflectError::new(kind, storage_path));
                    }
                    return Ok(self);
                }

                // Handle Map state transitions even when storing frames.
                // The Map needs to transition states so that subsequent operations work:
                // - PushingKey -> PushingValue: so begin_value() can be called
                // - PushingValue -> Idle: so begin_key() can be called for the next entry
                //
                // SSoT: we do NOT touch pending_entries here. The key/value buffer stays
                // owned by the stored frame. finish_deferred's walk will call
                // complete_map_{key,value}_frame at consume-time to transfer buffer
                // ownership into pending_entries *after* the frame has passed validation.
                if let Some(parent_frame) = stack.last_mut() {
                    if let Tracker::Map { insert_state, .. } = &mut parent_frame.tracker {
                        match insert_state {
                            MapInsertState::PushingKey { key_ptr, .. } => {
                                *insert_state = MapInsertState::PushingValue {
                                    key_ptr: *key_ptr,
                                    value_ptr: None,
                                };
                                crate::trace!(
                                    "end(): Map transitioned to PushingValue while storing key frame"
                                );
                            }
                            MapInsertState::PushingValue { .. } => {
                                *insert_state = MapInsertState::Idle;
                                crate::trace!(
                                    "end(): Map transitioned to Idle while storing value frame"
                                );
                            }
                            _ => {}
                        }
                    }

                    // Handle Set element insertion immediately.
                    // Set elements have no path identity (no index), so they can't be stored
                    // and re-entered. We must insert them into the Set now.
                    if let Tracker::Set { current_child } = &mut parent_frame.tracker
                        && *current_child
                        && parent_frame.is_init
                        && let Def::Set(set_def) = parent_frame.allocated.shape().def
                    {
                        let insert = set_def.vtable.insert;
                        let element_ptr = PtrMut::new(popped_frame.data.as_mut_byte_ptr());
                        unsafe {
                            insert(
                                PtrMut::new(parent_frame.data.as_mut_byte_ptr()),
                                element_ptr,
                            );
                        }
                        crate::trace!("end(): Set element inserted immediately in deferred mode");
                        // Insert moved out of popped_frame - don't store it
                        popped_frame.tracker = Tracker::Scalar;
                        popped_frame.is_init = false;
                        popped_frame.dealloc();
                        *current_child = false;
                        // Don't store this frame - return early
                        return Ok(self);
                    }

                    // Handle DynamicValue object entry - add to pending_entries for deferred insertion.
                    // Like Map entries, we store the key-value pair and insert during finalization.
                    if let Tracker::DynamicValue {
                        state:
                            DynamicValueState::Object {
                                insert_state,
                                pending_entries,
                            },
                    } = &mut parent_frame.tracker
                        && let DynamicObjectInsertState::BuildingValue { key } = insert_state
                    {
                        // Take ownership of the key from insert_state
                        let key = core::mem::take(key);

                        // Finalize the child Value before adding to pending_entries.
                        // The child might have its own pending_entries/pending_elements
                        // that need to be inserted first.
                        if let Err(kind) = popped_frame.require_full_initialization() {
                            return Err(ReflectError::new(kind, storage_path.clone()));
                        }

                        // Add to pending_entries for deferred insertion.
                        // The frame isn't stored — popped_frame is silently dropped after
                        // this block returns, and Frame has no Drop impl, so
                        // pending_entries is the sole owner of this buffer.
                        pending_entries.push((key, popped_frame.data));
                        crate::trace!(
                            "end(): DynamicValue object entry added to pending_entries in deferred mode"
                        );

                        // Reset insert state to Idle so more entries can be added
                        *insert_state = DynamicObjectInsertState::Idle;

                        // Don't store this frame - return early
                        return Ok(self);
                    }

                    // Handle DynamicValue array element - add to pending_elements for deferred insertion.
                    if let Tracker::DynamicValue {
                        state:
                            DynamicValueState::Array {
                                building_element,
                                pending_elements,
                            },
                    } = &mut parent_frame.tracker
                        && *building_element
                    {
                        // Finalize the child Value before adding to pending_elements.
                        // The child might have its own pending_entries/pending_elements
                        // that need to be inserted first.
                        if let Err(kind) = popped_frame.require_full_initialization() {
                            return Err(ReflectError::new(kind, storage_path.clone()));
                        }

                        // Add to pending_elements for deferred insertion.
                        // The frame isn't stored and Frame has no Drop impl, so
                        // pending_elements is the sole owner of this buffer.
                        pending_elements.push(popped_frame.data);
                        crate::trace!(
                            "end(): DynamicValue array element added to pending_elements in deferred mode"
                        );

                        // Reset building_element so more elements can be added
                        *building_element = false;

                        // Don't store this frame - return early
                        return Ok(self);
                    }

                    // Note: we intentionally do NOT call `rope.mark_last_initialized()`
                    // here for RopeSlot frames in deferred mode. Marking the slot as
                    // initialized now would let `ListRope::drain_into` drop it on a
                    // later error path — but the stored frame may have partial-init
                    // descendants (e.g. a Box<enum> whose fields weren't all set), and
                    // dropping those is UB. Instead, the frame stays stored; the
                    // `finish_deferred` walk marks the rope slot initialized only after
                    // `require_full_initialization` passes for this element. Same
                    // consume-time protocol used for Map `pending_entries`.

                    // Clear building_item for SmartPointerSlice so the next element can be added
                    if let Tracker::SmartPointerSlice { building_item, .. } =
                        &mut parent_frame.tracker
                    {
                        *building_item = false;
                        crate::trace!(
                            "end(): SmartPointerSlice building_item cleared while storing element"
                        );
                    }
                }

                stored_frames.insert(storage_path, popped_frame);

                // Clear parent's current_child tracking
                if let Some(parent_frame) = stack.last_mut() {
                    parent_frame.tracker.clear_current_child();
                }
            }

            return Ok(self);
        }

        // Update parent frame's tracking when popping from a child
        // Get parent shape upfront to avoid borrow conflicts
        let parent_shape = self.frames().last().unwrap().allocated.shape();
        let is_deferred_mode = self.is_deferred();
        let parent_frame = self.frames_mut().last_mut().unwrap();

        crate::trace!(
            "end(): Popped {} (tracker {:?}), Parent {} (tracker {:?})",
            popped_frame.allocated.shape(),
            popped_frame.tracker.kind(),
            parent_shape,
            parent_frame.tracker.kind()
        );

        // Check if we need to do a conversion - this happens when:
        // 1. The parent frame has a builder_shape or inner type that matches the popped frame's shape
        // 2. The parent frame has try_from
        // 3. The parent frame is not yet initialized
        // 4. The parent frame's tracker is Scalar or Inner (not Option, SmartPointer, etc.)
        //    This ensures we only do conversion when begin_inner was used, not begin_some
        let needs_conversion = !parent_frame.is_init
            && matches!(
                parent_frame.tracker,
                Tracker::Scalar | Tracker::Inner { .. }
            )
            && ((parent_shape.builder_shape.is_some()
                && parent_shape.builder_shape.unwrap() == popped_frame.allocated.shape())
                || (parent_shape.inner.is_some()
                    && parent_shape.inner.unwrap() == popped_frame.allocated.shape()))
            && match parent_shape.vtable {
                facet_core::VTableErased::Direct(vt) => vt.try_from.is_some(),
                facet_core::VTableErased::Indirect(vt) => vt.try_from.is_some(),
                // Unknown vtable kind: assume no try_from available.
                _ => false,
            };

        if needs_conversion {
            trace!(
                "Detected implicit conversion needed from {} to {}",
                popped_frame.allocated.shape(),
                parent_shape
            );

            // The conversion requires the source frame to be fully initialized
            // (we're about to call assume_init() and pass to try_from)
            if let Err(e) = popped_frame.require_full_initialization() {
                // Deallocate the memory since the frame wasn't fully initialized
                if let FrameOwnership::Owned = popped_frame.ownership
                    && let Ok(layout) = popped_frame.allocated.shape().layout.sized_layout()
                    && layout.size() > 0
                {
                    trace!(
                        "Deallocating uninitialized conversion frame memory: size={}, align={}",
                        layout.size(),
                        layout.align()
                    );
                    unsafe {
                        ::alloc::alloc::dealloc(popped_frame.data.as_mut_byte_ptr(), layout);
                    }
                }
                return Err(self.err(e));
            }

            // Perform the conversion
            let inner_ptr = unsafe { popped_frame.data.assume_init().as_const() };
            let inner_shape = popped_frame.allocated.shape();

            trace!("Converting from {} to {}", inner_shape, parent_shape);

            // Handle Direct and Indirect vtables - both return TryFromOutcome
            let outcome = match parent_shape.vtable {
                facet_core::VTableErased::Direct(vt) => {
                    if let Some(try_from_fn) = vt.try_from {
                        unsafe {
                            try_from_fn(
                                parent_frame.data.as_mut_byte_ptr() as *mut (),
                                inner_shape,
                                inner_ptr,
                            )
                        }
                    } else {
                        return Err(self.err(ReflectErrorKind::OperationFailed {
                            shape: parent_shape,
                            operation: "try_from not available for this type",
                        }));
                    }
                }
                facet_core::VTableErased::Indirect(vt) => {
                    if let Some(try_from_fn) = vt.try_from {
                        // parent_frame.data is uninitialized - we're writing the converted
                        // value into it
                        let ox_uninit =
                            facet_core::OxPtrUninit::new(parent_frame.data, parent_shape);
                        unsafe { try_from_fn(ox_uninit, inner_shape, inner_ptr) }
                    } else {
                        return Err(self.err(ReflectErrorKind::OperationFailed {
                            shape: parent_shape,
                            operation: "try_from not available for this type",
                        }));
                    }
                }
                // Unknown vtable kind: try_from not available for this type.
                _ => {
                    return Err(self.err(ReflectErrorKind::OperationFailed {
                        shape: parent_shape,
                        operation: "try_from not available for this type",
                    }));
                }
            };

            // Handle the TryFromOutcome, which explicitly communicates ownership semantics:
            // - Converted: source was consumed, conversion succeeded
            // - Unsupported: source was NOT consumed, caller retains ownership
            // - Failed: source WAS consumed, but conversion failed
            match outcome {
                facet_core::TryFromOutcome::Converted => {
                    trace!("Conversion succeeded, marking parent as initialized");
                    parent_frame.is_init = true;
                    // Reset Inner tracker to Scalar after successful conversion
                    if matches!(parent_frame.tracker, Tracker::Inner { .. }) {
                        parent_frame.tracker = Tracker::Scalar;
                    }
                }
                facet_core::TryFromOutcome::Unsupported => {
                    trace!("Source type not supported for conversion - source NOT consumed");

                    // Source was NOT consumed, so we need to drop it properly
                    if let FrameOwnership::Owned = popped_frame.ownership
                        && let Ok(layout) = popped_frame.allocated.shape().layout.sized_layout()
                        && layout.size() > 0
                    {
                        // Drop the value, then deallocate
                        unsafe {
                            popped_frame
                                .allocated
                                .shape()
                                .call_drop_in_place(popped_frame.data.assume_init());
                            ::alloc::alloc::dealloc(popped_frame.data.as_mut_byte_ptr(), layout);
                        }
                    }

                    return Err(self.err(ReflectErrorKind::TryFromError {
                        src_shape: inner_shape,
                        dst_shape: parent_shape,
                        inner: facet_core::TryFromError::UnsupportedSourceType,
                    }));
                }
                facet_core::TryFromOutcome::Failed(e) => {
                    trace!("Conversion failed after consuming source: {e:?}");

                    // Source WAS consumed, so we only deallocate memory (don't drop)
                    if let FrameOwnership::Owned = popped_frame.ownership
                        && let Ok(layout) = popped_frame.allocated.shape().layout.sized_layout()
                        && layout.size() > 0
                    {
                        trace!(
                            "Deallocating conversion frame memory after failure: size={}, align={}",
                            layout.size(),
                            layout.align()
                        );
                        unsafe {
                            ::alloc::alloc::dealloc(popped_frame.data.as_mut_byte_ptr(), layout);
                        }
                    }

                    return Err(self.err(ReflectErrorKind::TryFromError {
                        src_shape: inner_shape,
                        dst_shape: parent_shape,
                        inner: facet_core::TryFromError::Generic(e.into_owned()),
                    }));
                }
                // Unknown outcome: ownership semantics are unknown, so we conservatively
                // leave the source frame untouched (leaking rather than risking a
                // double-free) and report the conversion as failed.
                _ => {
                    return Err(self.err(ReflectErrorKind::TryFromError {
                        src_shape: inner_shape,
                        dst_shape: parent_shape,
                        inner: facet_core::TryFromError::UnsupportedSourceType,
                    }));
                }
            }

            // Deallocate the inner value's memory since try_from consumed it
            if let FrameOwnership::Owned = popped_frame.ownership
                && let Ok(layout) = popped_frame.allocated.shape().layout.sized_layout()
                && layout.size() > 0
            {
                trace!(
                    "Deallocating conversion frame memory: size={}, align={}",
                    layout.size(),
                    layout.align()
                );
                unsafe {
                    ::alloc::alloc::dealloc(popped_frame.data.as_mut_byte_ptr(), layout);
                }
            }

            return Ok(self);
        }

        // For Field-owned frames, reclaim responsibility in parent's tracker
        // Only mark as initialized if the child frame was actually initialized.
        // This prevents double-free when begin_inner/begin_some drops a value via
        // prepare_for_reinitialization but then fails, leaving the child uninitialized.
        //
        // We use require_full_initialization() rather than just is_init because:
        // - Scalar frames use is_init as the source of truth
        // - Struct/Array/Enum frames use their iset/data as the source of truth
        //   (is_init may never be set to true for these tracker types)
        if let FrameOwnership::Field { field_idx } = popped_frame.ownership {
            // In deferred mode, fill defaults on the child frame before checking initialization.
            // Fill defaults for child frame before checking if it's fully initialized.
            // This handles structs/enums with optional fields that should auto-fill.
            if let Err(e) = popped_frame.fill_defaults() {
                return Err(self.err(e));
            }
            let child_is_initialized = popped_frame.require_full_initialization().is_ok();
            match &mut parent_frame.tracker {
                Tracker::Struct {
                    iset,
                    current_child,
                } => {
                    if child_is_initialized {
                        iset.set(field_idx); // Parent reclaims responsibility only if child was init
                    }
                    *current_child = None;
                }
                Tracker::Array {
                    iset,
                    current_child,
                } => {
                    if child_is_initialized {
                        iset.set(field_idx); // Parent reclaims responsibility only if child was init
                    }
                    *current_child = None;
                }
                Tracker::Enum {
                    data,
                    current_child,
                    ..
                } => {
                    crate::trace!(
                        "end(): Enum field {} child_is_initialized={}, data before={:?}",
                        field_idx,
                        child_is_initialized,
                        data
                    );
                    if child_is_initialized {
                        data.set(field_idx); // Parent reclaims responsibility only if child was init
                    }
                    *current_child = None;
                }
                _ => {}
            }
            return Ok(self);
        }

        // For BorrowedInPlace DynamicValue frames (e.g., re-entered pending entries),
        // flush any pending_elements/pending_entries that were accumulated during
        // this re-entry. This is necessary because BorrowedInPlace frames aren't
        // stored for deferred processing - they modify existing memory in-place.
        if matches!(popped_frame.ownership, FrameOwnership::BorrowedInPlace)
            && let Err(e) = popped_frame.require_full_initialization()
        {
            return Err(self.err(e));
        }

        match &mut parent_frame.tracker {
            Tracker::SmartPointer {
                building_inner,
                pending_inner,
            } => {
                crate::trace!(
                    "end() SMARTPTR: popped {} into parent {} (building_inner={}, deferred={})",
                    popped_frame.allocated.shape(),
                    parent_frame.allocated.shape(),
                    *building_inner,
                    is_deferred_mode
                );
                // We just popped the inner value frame for a SmartPointer
                if *building_inner {
                    if matches!(parent_frame.allocated.shape().def, Def::Pointer(_)) {
                        // Check if we're in deferred mode - if so, retain the
                        // fully initialized inner staging allocation.
                        if is_deferred_mode {
                            if let Err(e) = popped_frame.fill_defaults() {
                                popped_frame.deinit();
                                popped_frame.dealloc();
                                return Err(self.err(e));
                            }
                            if let Err(e) = popped_frame.require_full_initialization() {
                                popped_frame.deinit();
                                popped_frame.dealloc();
                                return Err(self.err(e));
                            }

                            // `popped_frame` is not stored for re-entry. Transfer
                            // its exact staging metadata to the pending slot so
                            // finalization and cancellation use the right shape
                            // (for example, `String` rather than `str`).
                            *pending_inner = Some(
                                PendingSmartPointerInner::from_initialized_frame(popped_frame),
                            );
                            *building_inner = false;
                            // The parent pointer itself remains uninitialized
                            // until deferred finalization consumes this staging
                            // value.
                            parent_frame.is_init = false;
                            crate::trace!(
                                "end() SMARTPTR: stored pending_inner, will finalize in finish_deferred"
                            );
                        } else {
                            // Not in deferred mode - complete immediately
                            if let Def::Pointer(_) = parent_frame.allocated.shape().def {
                                if let Err(e) = popped_frame.require_full_initialization() {
                                    popped_frame.deinit();
                                    popped_frame.dealloc();
                                    return Err(self.err(e));
                                }

                                // Use complete_smart_pointer_frame which handles both:
                                // - Sized pointees (via owned transfer or borrow/promote)
                                // - Unsized pointees like str (via String conversion)
                                Self::complete_smart_pointer_frame(parent_frame, popped_frame);
                                crate::trace!(
                                    "end() SMARTPTR: completed smart pointer via complete_smart_pointer_frame"
                                );

                                // Change tracker to Scalar so the next end() just pops it
                                parent_frame.tracker = Tracker::Scalar;
                            }
                        }
                    } else {
                        return Err(self.err(ReflectErrorKind::OperationFailed {
                            shape: parent_shape,
                            operation: "SmartPointer frame without SmartPointer definition",
                        }));
                    }
                } else {
                    // building_inner is false - shouldn't happen in normal flow
                    return Err(self.err(ReflectErrorKind::OperationFailed {
                        shape: parent_shape,
                        operation: "SmartPointer end() called with building_inner = false",
                    }));
                }
            }
            Tracker::List {
                current_child,
                rope,
                ..
            } if parent_frame.is_init && current_child.is_some() => {
                // We just popped an element frame, now add it to the list
                if let Def::List(list_def) = parent_shape.def {
                    // Check which storage mode we used
                    if matches!(popped_frame.ownership, FrameOwnership::RopeSlot) {
                        // Rope storage: element lives in a stable chunk.
                        // Mark it as initialized; we'll drain to Vec when the list frame pops.
                        if let Some(rope) = rope {
                            rope.mark_last_initialized();
                        }
                        // No dealloc needed - memory belongs to rope
                    } else {
                        // Fallback: element is in separate heap buffer, use push to copy
                        let Some(push_fn) = list_def.push() else {
                            return Err(self.err(ReflectErrorKind::OperationFailed {
                                shape: parent_shape,
                                operation: "List missing push function",
                            }));
                        };

                        // The child frame contained the element value
                        let element_ptr = PtrMut::new(popped_frame.data.as_mut_byte_ptr());

                        // Use push to add element to the list
                        unsafe {
                            push_fn(
                                PtrMut::new(parent_frame.data.as_mut_byte_ptr()),
                                element_ptr,
                            );
                        }

                        // Push moved out of popped_frame
                        popped_frame.tracker = Tracker::Scalar;
                        popped_frame.is_init = false;
                        popped_frame.dealloc();
                    }

                    *current_child = None;
                }
            }
            Tracker::Map {
                insert_state,
                pending_entries,
                ..
            } if parent_frame.is_init => {
                match insert_state {
                    MapInsertState::PushingKey { key_ptr, .. } => {
                        // Fill defaults on the key frame before considering it done.
                        // This handles metadata containers and other structs with Option fields.
                        if let Err(e) = popped_frame.fill_defaults() {
                            return Err(self.err(e));
                        }

                        // Transfer key buffer ownership into pending_entries as a
                        // half-entry (key_ptr, None). popped_frame is silently dropped
                        // after this block (Frame has no Drop impl), so pending_entries
                        // becomes the sole owner. The value phase will upgrade the
                        // half-entry to a full (key, Some(value)) pair.
                        pending_entries.push((*key_ptr, None));

                        *insert_state = MapInsertState::PushingValue {
                            key_ptr: *key_ptr,
                            value_ptr: None,
                        };
                    }
                    MapInsertState::PushingValue { value_ptr, .. } => {
                        // Fill defaults on the value frame before considering it done.
                        // This handles structs with Option fields.
                        if let Err(e) = popped_frame.fill_defaults() {
                            return Err(self.err(e));
                        }

                        // Upgrade the last half-entry (key_ptr, None) to a full entry
                        // (key_ptr, Some(value_ptr)).
                        if let Some(value_ptr) = value_ptr {
                            let last = pending_entries.last_mut().expect(
                                "pending_entries must have a half-entry from the PushingKey -> PushingValue transition",
                            );
                            debug_assert!(
                                last.1.is_none(),
                                "last pending entry must be a half-entry (None value), got Some — invariant violation"
                            );
                            last.1 = Some(*value_ptr);

                            // Reset to idle state
                            *insert_state = MapInsertState::Idle;
                        }
                    }
                    MapInsertState::Idle => {
                        // Nothing to do
                    }
                }
            }
            Tracker::Set { current_child } if parent_frame.is_init && *current_child => {
                // We just popped an element frame, now insert it into the set
                if let Def::Set(set_def) = parent_frame.allocated.shape().def {
                    let insert = set_def.vtable.insert;

                    // The child frame contained the element value
                    let element_ptr = PtrMut::new(popped_frame.data.as_mut_byte_ptr());

                    // Use insert to add element to the set
                    unsafe {
                        insert(
                            PtrMut::new(parent_frame.data.as_mut_byte_ptr()),
                            element_ptr,
                        );
                    }

                    // Insert moved out of popped_frame
                    popped_frame.tracker = Tracker::Scalar;
                    popped_frame.is_init = false;
                    popped_frame.dealloc();

                    *current_child = false;
                }
            }
            Tracker::Option {
                building_inner,
                pending_inner,
            } => {
                crate::trace!(
                    "end(): matched Tracker::Option, building_inner={}",
                    *building_inner
                );
                // We just popped the inner value frame for an Option's Some variant
                if *building_inner {
                    if matches!(parent_frame.allocated.shape().def, Def::Option(_)) {
                        // Store the inner value pointer for deferred init_some.
                        // This keeps the inner value's memory stable for deferred processing.
                        // Actual init_some() happens in require_full_initialization().
                        //
                        // popped_frame isn't stored — it's silently dropped after this
                        // block (Frame has no Drop impl), so pending_inner is the sole
                        // owner of this buffer.
                        *pending_inner = Some(popped_frame.data);

                        // Mark that we're no longer building the inner value
                        *building_inner = false;
                        crate::trace!("end(): stored pending_inner, set building_inner to false");
                        // Mark the Option as initialized (pending finalization)
                        parent_frame.is_init = true;
                        crate::trace!("end(): set parent_frame.is_init to true");
                    } else {
                        return Err(self.err(ReflectErrorKind::OperationFailed {
                            shape: parent_shape,
                            operation: "Option frame without Option definition",
                        }));
                    }
                } else {
                    // building_inner is false - the Option was already initialized but
                    // begin_some was called again. The popped frame was not used to
                    // initialize the Option, so we need to clean it up.
                    popped_frame.deinit();
                    if let FrameOwnership::Owned = popped_frame.ownership
                        && let Ok(layout) = popped_frame.allocated.shape().layout.sized_layout()
                        && layout.size() > 0
                    {
                        unsafe {
                            ::alloc::alloc::dealloc(popped_frame.data.as_mut_byte_ptr(), layout);
                        }
                    }
                }
            }
            Tracker::Result {
                is_ok,
                building_inner,
            } => {
                crate::trace!(
                    "end(): matched Tracker::Result, is_ok={}, building_inner={}",
                    *is_ok,
                    *building_inner
                );
                // We just popped the inner value frame for a Result's Ok or Err variant
                if *building_inner {
                    if let Def::Result(result_def) = parent_frame.allocated.shape().def {
                        // The popped frame contains the inner value
                        let inner_value_ptr = unsafe { popped_frame.data.assume_init() };

                        // Initialize the Result as Ok(inner_value) or Err(inner_value)
                        if *is_ok {
                            let init_ok_fn = result_def.vtable.init_ok;
                            unsafe {
                                init_ok_fn(parent_frame.data, inner_value_ptr);
                            }
                        } else {
                            let init_err_fn = result_def.vtable.init_err;
                            unsafe {
                                init_err_fn(parent_frame.data, inner_value_ptr);
                            }
                        }

                        // Deallocate the inner value's memory since init_ok/err_fn moved it
                        if let FrameOwnership::Owned = popped_frame.ownership
                            && let Ok(layout) = popped_frame.allocated.shape().layout.sized_layout()
                            && layout.size() > 0
                        {
                            unsafe {
                                ::alloc::alloc::dealloc(
                                    popped_frame.data.as_mut_byte_ptr(),
                                    layout,
                                );
                            }
                        }

                        // Mark that we're no longer building the inner value
                        *building_inner = false;
                        crate::trace!("end(): set building_inner to false");
                        // Mark the Result as initialized
                        parent_frame.is_init = true;
                        crate::trace!("end(): set parent_frame.is_init to true");
                    } else {
                        return Err(self.err(ReflectErrorKind::OperationFailed {
                            shape: parent_shape,
                            operation: "Result frame without Result definition",
                        }));
                    }
                } else {
                    // building_inner is false - the Result was already initialized but
                    // begin_ok/begin_err was called again. The popped frame was not used to
                    // initialize the Result, so we need to clean it up.
                    popped_frame.deinit();
                    if let FrameOwnership::Owned = popped_frame.ownership
                        && let Ok(layout) = popped_frame.allocated.shape().layout.sized_layout()
                        && layout.size() > 0
                    {
                        unsafe {
                            ::alloc::alloc::dealloc(popped_frame.data.as_mut_byte_ptr(), layout);
                        }
                    }
                }
            }
            Tracker::Scalar => {
                // the main case here is: the popped frame was a `String` and the
                // parent frame is an `Arc<str>`, `Box<str>` etc.
                match &parent_shape.def {
                    Def::Pointer(smart_ptr_def) => {
                        let pointee = match smart_ptr_def.pointee() {
                            Some(p) => p,
                            None => {
                                return Err(self.err(ReflectErrorKind::InvariantViolation {
                                    invariant: "pointer type doesn't have a pointee",
                                }));
                            }
                        };

                        if !pointee.is_shape(str::SHAPE) {
                            return Err(self.err(ReflectErrorKind::InvariantViolation {
                                invariant: "only T=str is supported when building SmartPointer<T> and T is unsized",
                            }));
                        }

                        if !popped_frame.allocated.shape().is_shape(String::SHAPE) {
                            return Err(self.err(ReflectErrorKind::InvariantViolation {
                                invariant: "the popped frame should be String when building a SmartPointer<T>",
                            }));
                        }

                        if let Err(e) = popped_frame.require_full_initialization() {
                            return Err(self.err(e));
                        }

                        // if the just-popped frame was a SmartPointerStr, we have some conversion to do:
                        // Special-case: SmartPointer<str> (Box<str>, Arc<str>, Rc<str>) via SmartPointerStr tracker
                        // Here, popped_frame actually contains a value for String that should be moved into the smart pointer.
                        // We convert the String into Box<str>, Arc<str>, or Rc<str> as appropriate and write it to the parent frame.
                        use ::alloc::{rc::Rc, string::String, sync::Arc};

                        let Some(known) = smart_ptr_def.known else {
                            return Err(self.err(ReflectErrorKind::OperationFailed {
                                shape: parent_shape,
                                operation: "SmartPointerStr for unknown smart pointer kind",
                            }));
                        };

                        parent_frame.deinit();

                        // Interpret the memory as a String, then convert and write.
                        let string_ptr = popped_frame.data.as_mut_byte_ptr() as *mut String;
                        let string_value = unsafe { core::ptr::read(string_ptr) };

                        match known {
                            KnownPointer::Box => {
                                let boxed: Box<str> = string_value.into_boxed_str();
                                unsafe {
                                    core::ptr::write(
                                        parent_frame.data.as_mut_byte_ptr() as *mut Box<str>,
                                        boxed,
                                    );
                                }
                            }
                            KnownPointer::Arc => {
                                let arc: Arc<str> = Arc::from(string_value.into_boxed_str());
                                unsafe {
                                    core::ptr::write(
                                        parent_frame.data.as_mut_byte_ptr() as *mut Arc<str>,
                                        arc,
                                    );
                                }
                            }
                            KnownPointer::Rc => {
                                let rc: Rc<str> = Rc::from(string_value.into_boxed_str());
                                unsafe {
                                    core::ptr::write(
                                        parent_frame.data.as_mut_byte_ptr() as *mut Rc<str>,
                                        rc,
                                    );
                                }
                            }
                            _ => {
                                return Err(self.err(ReflectErrorKind::OperationFailed {
                                    shape: parent_shape,
                                    operation: "Don't know how to build this pointer type",
                                }));
                            }
                        }

                        parent_frame.is_init = true;

                        popped_frame.tracker = Tracker::Scalar;
                        popped_frame.is_init = false;
                        popped_frame.dealloc();
                    }
                    _ => {
                        // This can happen if begin_inner() was called on a type that
                        // has shape.inner but isn't a SmartPointer (e.g., Option).
                        // In this case, we can't complete the conversion, so return error.
                        return Err(self.err(ReflectErrorKind::OperationFailed {
                            shape: parent_shape,
                            operation: "end() called but parent has Uninit/Init tracker and isn't a SmartPointer",
                        }));
                    }
                }
            }
            Tracker::SmartPointerSlice {
                vtable,
                building_item,
                ..
            } if *building_item => {
                // We just popped an element frame, now push it to the slice builder
                let element_ptr = PtrMut::new(popped_frame.data.as_mut_byte_ptr());

                // Use the slice builder's push_fn to add the element
                crate::trace!("Pushing element to slice builder");
                unsafe {
                    let parent_ptr = parent_frame.data.assume_init();
                    (vtable.push_fn)(parent_ptr, element_ptr);
                }

                popped_frame.tracker = Tracker::Scalar;
                popped_frame.is_init = false;
                popped_frame.dealloc();

                if let Tracker::SmartPointerSlice {
                    building_item: bi, ..
                } = &mut parent_frame.tracker
                {
                    *bi = false;
                }
            }
            Tracker::DynamicValue {
                state:
                    DynamicValueState::Array {
                        building_element, ..
                    },
            } if *building_element => {
                // Check that the element is initialized before pushing
                if !popped_frame.is_init {
                    // Element was never set - clean up and return error
                    let shape = parent_frame.allocated.shape();
                    popped_frame.dealloc();
                    *building_element = false;
                    // No need to poison - returning Err consumes self, Drop will handle cleanup
                    return Err(self.err(ReflectErrorKind::OperationFailed {
                        shape,
                        operation: "end() called but array element was never initialized",
                    }));
                }

                // We just popped an element frame, now push it to the dynamic array
                if let Def::DynamicValue(dyn_def) = parent_frame.allocated.shape().def {
                    // Get mutable pointers - both array and element need PtrMut
                    let array_ptr = unsafe { parent_frame.data.assume_init() };
                    let element_ptr = unsafe { popped_frame.data.assume_init() };

                    // Use push_array_element to add element to the array
                    unsafe {
                        (dyn_def.vtable.push_array_element)(array_ptr, element_ptr);
                    }

                    // Push moved out of popped_frame
                    popped_frame.tracker = Tracker::Scalar;
                    popped_frame.is_init = false;
                    popped_frame.dealloc();

                    *building_element = false;
                }
            }
            Tracker::DynamicValue {
                state: DynamicValueState::Object { insert_state, .. },
            } => {
                if let DynamicObjectInsertState::BuildingValue { key } = insert_state {
                    // Check that the value is initialized before inserting
                    if !popped_frame.is_init {
                        // Value was never set - clean up and return error
                        let shape = parent_frame.allocated.shape();
                        popped_frame.dealloc();
                        *insert_state = DynamicObjectInsertState::Idle;
                        // No need to poison - returning Err consumes self, Drop will handle cleanup
                        return Err(self.err(ReflectErrorKind::OperationFailed {
                            shape,
                            operation: "end() called but object entry value was never initialized",
                        }));
                    }

                    // We just popped a value frame, now insert it into the dynamic object
                    if let Def::DynamicValue(dyn_def) = parent_frame.allocated.shape().def {
                        // Get mutable pointers - both object and value need PtrMut
                        let object_ptr = unsafe { parent_frame.data.assume_init() };
                        let value_ptr = unsafe { popped_frame.data.assume_init() };

                        // Use insert_object_entry to add the key-value pair
                        unsafe {
                            (dyn_def.vtable.insert_object_entry)(object_ptr, key, value_ptr);
                        }

                        // Insert moved out of popped_frame
                        popped_frame.tracker = Tracker::Scalar;
                        popped_frame.is_init = false;
                        popped_frame.dealloc();

                        // Reset insert state to Idle
                        *insert_state = DynamicObjectInsertState::Idle;
                    }
                }
            }
            _ => {}
        }

        Ok(self)
    }

    /// Returns a path representing the current traversal in the builder.
    ///
    /// The returned [`facet_path::Path`] can be formatted as a human-readable string
    /// using [`Path::format_with_shape()`](facet_path::Path::format_with_shape),
    /// e.g., `fieldName[index].subfield`.
    pub fn path(&self) -> Path {
        use facet_path::PathStep;

        let root_shape = self
            .frames()
            .first()
            .expect("Partial must have at least one frame")
            .allocated
            .shape();
        let mut path = Path::new(root_shape);

        for frame in self.frames().iter() {
            match frame.allocated.shape().ty {
                Type::User(user_type) => match user_type {
                    UserType::Struct(_struct_type) => {
                        // Add field step if we're currently in a field
                        if let Tracker::Struct {
                            current_child: Some(idx),
                            ..
                        } = &frame.tracker
                        {
                            path.push(PathStep::Field(*idx as u32));
                        }
                    }
                    UserType::Enum(enum_type) => {
                        // Add variant and optional field step
                        if let Tracker::Enum {
                            variant,
                            current_child,
                            ..
                        } = &frame.tracker
                        {
                            // Find the variant index by comparing pointers
                            if let Some(variant_idx) = enum_type
                                .variants
                                .iter()
                                .position(|v| core::ptr::eq(v, *variant))
                            {
                                path.push(PathStep::Variant(variant_idx as u32));
                            }
                            if let Some(idx) = *current_child {
                                path.push(PathStep::Field(idx as u32));
                            }
                        }
                    }
                    UserType::Union(_) => {
                        // No structural path steps for unions
                    }
                    UserType::Opaque => {
                        // Opaque types might be lists (e.g., Vec<T>)
                        if let Tracker::List {
                            current_child: Some(idx),
                            ..
                        } = &frame.tracker
                        {
                            path.push(PathStep::Index(*idx as u32));
                        }
                    }
                    _ => {
                        // Unknown user types contribute no structural path steps.
                    }
                },
                Type::Sequence(facet_core::SequenceType::Array(_array_def)) => {
                    // Add index step if we're currently in an element
                    if let Tracker::Array {
                        current_child: Some(idx),
                        ..
                    } = &frame.tracker
                    {
                        path.push(PathStep::Index(*idx as u32));
                    }
                }
                Type::Sequence(_) => {
                    // Other sequence types (Slice, etc.) - no index tracking
                }
                Type::Pointer(_) => {
                    path.push(PathStep::Deref);
                }
                _ => {
                    // No structural path for scalars, etc.
                }
            }
        }

        path
    }

    /// Returns the root shape for path formatting.
    ///
    /// Use this together with [`path()`](Self::path) to format the path:
    /// ```ignore
    /// let path_str = partial.path().format_with_shape(partial.root_shape());
    /// ```
    pub fn root_shape(&self) -> &'static Shape {
        self.frames()
            .first()
            .expect("Partial should always have at least one frame")
            .allocated
            .shape()
    }

    /// Create a [`ReflectError`] with the current path context.
    ///
    /// This is a convenience method for constructing errors inside `Partial` methods
    /// that automatically captures the current traversal path.
    #[inline]
    pub fn err(&self, kind: ReflectErrorKind) -> ReflectError {
        ReflectError::new(kind, self.path())
    }

    /// Get the field for the parent frame
    pub fn parent_field(&self) -> Option<&Field> {
        self.frames()
            .iter()
            .rev()
            .nth(1)
            .and_then(|f| f.get_field())
    }

    /// Gets the field for the current frame
    pub fn current_field(&self) -> Option<&Field> {
        self.frames().last().and_then(|f| f.get_field())
    }

    /// Gets the nearest active field when nested wrapper frames are involved.
    ///
    /// This walks frames from innermost to outermost and returns the first frame
    /// that currently points at a struct/enum field.
    pub fn nearest_field(&self) -> Option<&Field> {
        self.frames().iter().rev().find_map(|f| f.get_field())
    }

    /// Returns a const pointer to the current frame's data.
    ///
    /// This is useful for validation - after deserializing a field value,
    /// validators can read the value through this pointer.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid only while the frame exists.
    /// The caller must ensure the frame is fully initialized before
    /// reading through this pointer.
    #[deprecated(note = "use initialized_data_ptr() instead, which checks initialization")]
    pub fn data_ptr(&self) -> Option<facet_core::PtrConst> {
        if self.state != PartialState::Active {
            return None;
        }
        self.frames().last().map(|f| {
            // SAFETY: We're in active state, so the frame is valid.
            // The caller is responsible for ensuring the data is initialized.
            unsafe { f.data.assume_init().as_const() }
        })
    }

    /// Returns a const pointer to the current frame's data, but only if fully initialized.
    ///
    /// This is the safe way to get a pointer for validation - it verifies that
    /// the frame is fully initialized before returning the pointer.
    ///
    /// Returns `None` if:
    /// - The partial is not in active state
    /// - The current frame is not fully initialized
    #[allow(unsafe_code)]
    pub fn initialized_data_ptr(&mut self) -> Option<facet_core::PtrConst> {
        if self.state != PartialState::Active {
            return None;
        }
        let frame = self.frames_mut().last_mut()?;

        // Check if fully initialized (may drain rope for lists)
        if frame.require_full_initialization().is_err() {
            return None;
        }

        // SAFETY: We've verified the partial is active and the frame is fully initialized.
        Some(unsafe { frame.data.assume_init().as_const() })
    }

    /// Returns a typed reference to the current frame's data if:
    /// 1. The partial is in active state
    /// 2. The current frame is fully initialized
    /// 3. The shape matches `T::SHAPE`
    ///
    /// This is the safe way to read a value from a Partial for validation purposes.
    #[allow(unsafe_code)]
    pub fn read_as<T: facet_core::Facet<'facet>>(&mut self) -> Option<&T> {
        if self.state != PartialState::Active {
            return None;
        }
        let frame = self.frames_mut().last_mut()?;

        // Check if fully initialized (may drain rope for lists)
        if frame.require_full_initialization().is_err() {
            return None;
        }

        // Check shape matches
        if frame.allocated.shape() != T::SHAPE {
            return None;
        }

        // SAFETY: We've verified:
        // 1. The partial is active (frame is valid)
        // 2. The frame is fully initialized
        // 3. The shape matches T::SHAPE
        unsafe {
            let ptr = frame.data.assume_init().as_const();
            Some(&*ptr.as_ptr::<T>())
        }
    }
}
