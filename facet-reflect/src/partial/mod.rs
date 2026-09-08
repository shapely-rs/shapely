//! Partial value construction for dynamic reflection
//!
//! This module provides APIs for incrementally building values through reflection,
//! particularly useful when deserializing data from external formats like JSON or YAML.
//!
//! # Overview
//!
//! The `Partial` type (formerly known as `Wip` - Work In Progress) allows you to:
//! - Allocate memory for a value based on its `Shape`
//! - Initialize fields incrementally in a type-safe manner
//! - Handle complex nested structures including structs, enums, collections, and smart pointers
//! - Build the final value once all required fields are initialized
//!
//! **Note**: This is the only API for partial value construction. The previous `TypedPartial`
//! wrapper has been removed in favor of using `Partial` directly.
//!
//! # Basic Usage
//!
//! ```no_run
//! # use facet_reflect::Partial;
//! # use facet_core::{Shape, Facet};
//! # fn example<T: Facet<'static>>() -> Result<(), Box<dyn std::error::Error>> {
//! // Allocate memory for a struct
//! let mut partial = Partial::alloc::<T>()?;
//!
//! // Set simple fields
//! partial = partial.set_field("name", "Alice")?;
//! partial = partial.set_field("age", 30u32)?;
//!
//! // Work with nested structures
//! partial = partial.begin_field("address")?;
//! partial = partial.set_field("street", "123 Main St")?;
//! partial = partial.set_field("city", "Springfield")?;
//! partial = partial.end()?;
//!
//! // Build the final value
//! let value = partial.build()?;
//! # Ok(())
//! # }
//! ```
//!
//! # Chaining Style
//!
//! The API supports method chaining for cleaner code:
//!
//! ```no_run
//! # use facet_reflect::Partial;
//! # use facet_core::{Shape, Facet};
//! # fn example<T: Facet<'static>>() -> Result<(), Box<dyn std::error::Error>> {
//! let value = Partial::alloc::<T>()?
//!     .set_field("name", "Bob")?
//!     .begin_field("scores")?
//!         .set(vec![95, 87, 92])?
//!     .end()?
//!     .build()?;
//! # Ok(())
//! # }
//! ```
//!
//! # Working with Collections
//!
//! ```no_run
//! # use facet_reflect::Partial;
//! # use facet_core::{Shape, Facet};
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut partial = Partial::alloc::<Vec<String>>()?;
//!
//! // Add items to a list
//! partial = partial.begin_list_item()?;
//! partial = partial.set("first")?;
//! partial = partial.end()?;
//!
//! partial = partial.begin_list_item()?;
//! partial = partial.set("second")?;
//! partial = partial.end()?;
//!
//! let vec = partial.build()?;
//! # Ok(())
//! # }
//! ```
//!
//! # Working with Maps
//!
//! ```no_run
//! # use facet_reflect::Partial;
//! # use facet_core::{Shape, Facet};
//! # use std::collections::HashMap;
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut partial = Partial::alloc::<HashMap<String, i32>>()?;
//!
//! // Insert key-value pairs
//! partial = partial.begin_key()?;
//! partial = partial.set("score")?;
//! partial = partial.end()?;
//! partial = partial.begin_value()?;
//! partial = partial.set(100i32)?;
//! partial = partial.end()?;
//!
//! let map = partial.build()?;
//! # Ok(())
//! # }
//! ```
//!
//! # Safety and Memory Management
//!
//! The `Partial` type ensures memory safety by:
//! - Tracking initialization state of all fields
//! - Preventing use-after-build through state tracking
//! - Properly handling drop semantics for partially initialized values
//! - Supporting both owned and borrowed values through lifetime parameters
//!
//! # Drop-ownership invariant (single-source-of-truth)
//!
//! Every heap buffer allocated during partial construction has exactly one drop-and-dealloc
//! authority at all times. Authority is either the owning frame (its `FrameOwnership`
//! controls deinit + dealloc) or a parent tracker's pending slot
//! (`Tracker::Map::pending_entries`, `Tracker::Option::pending_inner`,
//! `Tracker::SmartPointer::pending_inner`, `DynamicValueState::Object::pending_entries`,
//! `DynamicValueState::Array::pending_elements`). Authority transfers between the two
//! at well-defined points — never simultaneously held.
//!
//! - **Stored frames in deferred mode keep their buffer**: a frame that gets stored in
//!   `stored_frames` for later re-entry retains full `TrackedBuffer`/`Owned` ownership
//!   of its data. Parent tracker pending slots are NOT populated at store-time. On
//!   error (`Partial` dropped mid-build, or `finish_deferred` aborting), the stored
//!   frame's own `deinit` + `dealloc` handles cleanup; no parent pending-slot severing
//!   is required.
//! - **Pending-slot population is consume-time**: `finish_deferred`'s walk calls
//!   `complete_map_{key,value}_frame` / `complete_option_frame` /
//!   `complete_smart_pointer_frame` AFTER `require_full_initialization` has validated
//!   the child frame. Only then does the buffer pointer move into the parent's pending
//!   slot (or directly into the parent's final storage, for Option/SmartPointer). The
//!   child frame is dropped silently (Frame has no `Drop` impl); the parent's pending
//!   slot becomes the sole owner.
//! - **Non-stored frames in deferred mode**: when a child frame isn't stored (e.g.
//!   scalar Option inner), its popped `data` pointer is moved into the parent pending
//!   slot directly in `end()`. The frame is silently dropped after the match block, so
//!   single ownership is preserved without any ownership mutation.
//! - **Map half-entries**: `pending_entries` holds `(key_ptr, Option<value_ptr>)`. A key
//!   pushed without a paired value is a half-entry `(key, None)`; its value-phase
//!   counterpart upgrades the last entry to `(key, Some(value))`. A half-entry left at
//!   finalize is an invariant violation; a half-entry at drop-time drops the orphan key
//!   only.
//! - **No sever helpers, no ownership-mutation protocol**: the walk's consume-time
//!   ordering guarantees `pending_entries` / `pending_inner` entries only reference
//!   fully-validated buffers. `Tracker::*::deinit` can drain these unconditionally.
//!   If you catch yourself adding a `sever_parent_pending_*` or `untransfer_*` helper,
//!   re-read this block — the answer is to push to the pending slot later, not to add
//!   a sever step.

use alloc::{collections::BTreeMap, sync::Arc, vec::Vec};

mod arena;
mod iset;
mod rope;
pub(crate) mod typeplan;
pub use typeplan::{
    DeserStrategy, EnumPlan, FieldDefault, FieldPlan, FillRule, NodeId, StructPlan, TypePlan,
    TypePlanCore, TypePlanNode, TypePlanNodeKind, VariantPlanMeta,
};

mod partial_api;

use crate::{ReflectErrorKind, TrackerKind, trace};
use facet_core::Facet;
use facet_path::{Path, PathStep};

use core::marker::PhantomData;

mod heap_value;
pub use heap_value::*;

use facet_core::{
    Def, EnumType, Field, PtrMut, PtrUninit, Shape, SliceBuilderVTable, Type, UserType, Variant,
};
use iset::ISet;
use rope::ListRope;
use typeplan::FieldInitPlan;

/// State of a partial value
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PartialState {
    /// Partial is active and can be modified
    Active,

    /// Partial has been successfully built and cannot be reused
    Built,
}

/// Mode of operation for frame management.
///
/// In `Strict` mode, frames must be fully initialized before being popped.
/// In `Deferred` mode, frames can be stored when popped and restored on re-entry,
/// with final validation happening in `finish_deferred()`.
enum FrameMode {
    /// Strict mode: frames must be fully initialized before popping.
    Strict {
        /// Stack of frames for nested initialization.
        stack: Vec<Frame>,
    },

    /// Deferred mode: frames are stored when popped, can be re-entered.
    Deferred {
        /// Stack of frames for nested initialization.
        stack: Vec<Frame>,

        /// The frame depth when deferred mode was started.
        /// Path calculations are relative to this depth.
        start_depth: usize,

        /// Frames saved when popped, keyed by their path (derived from frame stack).
        /// When we re-enter a path, we restore the stored frame.
        /// Uses the full `Path` type which includes the root shape for proper type anchoring.
        stored_frames: BTreeMap<Path, Frame>,
    },
}

impl FrameMode {
    /// Get a reference to the frame stack.
    const fn stack(&self) -> &Vec<Frame> {
        match self {
            FrameMode::Strict { stack } | FrameMode::Deferred { stack, .. } => stack,
        }
    }

    /// Get a mutable reference to the frame stack.
    const fn stack_mut(&mut self) -> &mut Vec<Frame> {
        match self {
            FrameMode::Strict { stack } | FrameMode::Deferred { stack, .. } => stack,
        }
    }

    /// Check if we're in deferred mode.
    const fn is_deferred(&self) -> bool {
        matches!(self, FrameMode::Deferred { .. })
    }

    /// Get the start depth if in deferred mode.
    const fn start_depth(&self) -> Option<usize> {
        match self {
            FrameMode::Deferred { start_depth, .. } => Some(*start_depth),
            FrameMode::Strict { .. } => None,
        }
    }
}

/// A type-erased, heap-allocated, partially-initialized value.
///
/// [Partial] keeps track of the state of initialiation of the underlying
/// value: if we're building `struct S { a: u32, b: String }`, we may
/// have initialized `a`, or `b`, or both, or neither.
///
/// [Partial] allows navigating down nested structs and initializing them
/// progressively: [Partial::begin_field] pushes a frame onto the stack,
/// which then has to be initialized, and popped off with [Partial::end].
///
/// If [Partial::end] is called but the current frame isn't fully initialized,
/// an error is returned: in other words, if you navigate down to a field,
/// you have to fully initialize it one go. You can't go back up and back down
/// to it again.
pub struct Partial<'facet, const BORROW: bool = true> {
    /// Frame management mode (strict or deferred) and associated state.
    mode: FrameMode,

    /// current state of the Partial
    state: PartialState,

    /// Precomputed deserialization plan for the root type.
    /// Built once at allocation time, navigated in parallel with value construction.
    /// Each Frame holds a NodeId (index) into this plan's arenas.
    root_plan: Arc<TypePlanCore>,

    /// PhantomData marker for the 'facet lifetime.
    /// This is covariant in 'facet, which is safe because 'facet represents
    /// the lifetime of borrowed data FROM the input (deserialization source).
    /// A Partial<'long, ...> can be safely treated as Partial<'short, ...>
    /// because it only needs borrowed data to live at least as long as 'short.
    _marker: PhantomData<&'facet ()>,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum MapInsertState {
    /// Not currently inserting
    Idle,

    /// Pushing key - memory allocated, waiting for initialization.
    ///
    /// The key buffer is owned by the key frame currently on the stack
    /// (a `TrackedBuffer` frame). In non-deferred mode, the key frame's `end()`
    /// transfers the buffer into `pending_entries` as a half-entry
    /// `(key_ptr, None)` before transitioning to `PushingValue`. In deferred
    /// mode, the key frame is stored (retaining ownership) and only the state
    /// transitions here; `pending_entries` is populated at consume-time by
    /// `complete_map_key_frame` during `finish_deferred`.
    PushingKey {
        /// Temporary storage for the key being built
        key_ptr: PtrUninit,
    },

    /// Pushing value after key is done.
    ///
    /// In non-deferred mode, the value frame's `end()` upgrades the last
    /// half-entry in `pending_entries` to `(key_ptr, Some(value_ptr))`. In
    /// deferred mode, the value frame is stored (retaining ownership); the
    /// `finish_deferred` walk's `complete_map_value_frame` upgrades
    /// the half-entry at consume-time.
    PushingValue {
        /// Temporary storage for the key that was built (always initialized)
        key_ptr: PtrUninit,
        /// Temporary storage for the value being built
        value_ptr: Option<PtrUninit>,
    },
}

#[derive(Debug, Clone, Copy)]
pub(crate) enum FrameOwnership {
    /// This frame owns the allocation and should deallocate it on drop
    Owned,

    /// This frame points to a field/element within a parent's allocation.
    /// The parent's `iset[field_idx]` was CLEARED when this frame was created.
    /// On drop: deinit if initialized, but do NOT deallocate.
    /// On successful end(): parent's `iset[field_idx]` will be SET.
    Field { field_idx: usize },

    /// Temporary buffer tracked by parent's MapInsertState.
    /// Used by begin_key(), begin_value() for map insertions.
    /// Safe to drop on deinit - parent's cleanup respects is_init propagation.
    TrackedBuffer,

    /// Pointer into existing collection entry (Value object, Option inner, etc.)
    /// Used by begin_object_entry() on existing key, begin_some() re-entry.
    /// NOT safe to drop on deinit - parent collection has no per-entry tracking
    /// and would try to drop the freed value again (double-free).
    BorrowedInPlace,

    /// Pointer to externally-owned memory (e.g., caller's stack via MaybeUninit).
    /// Used by `from_raw()` for stack-friendly deserialization.
    /// On drop: deinit if initialized (drop partially constructed values), but do NOT deallocate.
    /// The caller owns the memory and is responsible for its lifetime.
    External,

    /// Points into a stable rope chunk for list element building.
    /// Used by `begin_list_item()` for building Vec elements.
    /// The memory is stable (won't move during Vec growth),
    /// so frames inside can be stored for deferred processing.
    /// On successful end(): element is tracked for later finalization.
    /// On list frame end(): all elements are moved into the real Vec.
    /// On drop/failure: the rope chunk handles cleanup.
    RopeSlot,
}

impl FrameOwnership {
    /// Returns true if this frame is responsible for deallocating its memory.
    ///
    /// Both `Owned` and `TrackedBuffer` frames allocated their memory and need
    /// to deallocate it. `Field`, `BorrowedInPlace`, `External`, and `RopeSlot`
    /// frames borrow from somewhere else.
    const fn needs_dealloc(&self) -> bool {
        matches!(self, FrameOwnership::Owned | FrameOwnership::TrackedBuffer)
    }
}

/// Immutable pairing of a shape with its actual allocation size.
///
/// This ensures that the shape and allocated size are always in sync and cannot
/// drift apart, preventing the class of bugs where a frame's shape doesn't match
/// what was actually allocated (see issue #1568).
#[derive(Debug)]
pub(crate) struct AllocatedShape {
    shape: &'static Shape,
    allocated_size: usize,
}

impl AllocatedShape {
    pub(crate) const fn new(shape: &'static Shape, allocated_size: usize) -> Self {
        Self {
            shape,
            allocated_size,
        }
    }

    pub(crate) const fn shape(&self) -> &'static Shape {
        self.shape
    }

    pub(crate) const fn allocated_size(&self) -> usize {
        self.allocated_size
    }

    /// Deallocate a buffer allocated for this shape's exact recorded size.
    ///
    /// # Safety
    ///
    /// `data` must be the still-valid allocation paired with this shape.
    unsafe fn deallocate(&self, data: PtrUninit) {
        if self.allocated_size() == 0 {
            return;
        }

        if let Ok(layout) = self.shape.layout.sized_layout() {
            let actual_layout =
                core::alloc::Layout::from_size_align(self.allocated_size(), layout.align())
                    .expect("allocated_size must be valid");
            unsafe { alloc::alloc::dealloc(data.as_mut_byte_ptr(), actual_layout) };
        }
    }
}

/// Points somewhere in a partially-initialized value. If we're initializing
/// `a.b.c`, then the first frame would point to the beginning of `a`, the
/// second to the beginning of the `b` field of `a`, etc.
///
/// A frame can point to a complex data structure, like a struct or an enum:
/// it keeps track of whether a variant was selected, which fields are initialized,
/// etc. and is able to drop & deinitialize
#[must_use]
pub(crate) struct Frame {
    /// Address of the value being initialized
    pub(crate) data: PtrUninit,

    /// Shape of the value being initialized, paired with the actual allocation size
    pub(crate) allocated: AllocatedShape,

    /// Whether this frame's data is fully initialized
    pub(crate) is_init: bool,

    /// Tracks building mode and partial initialization state
    pub(crate) tracker: Tracker,

    /// Whether this frame owns the allocation or is just a field pointer
    pub(crate) ownership: FrameOwnership,

    /// Whether this frame is for a custom deserialization pipeline
    pub(crate) using_custom_deserialization: bool,

    /// Container-level proxy definition (from `#[facet(proxy = ...)]` on the shape).
    /// Used during custom deserialization to convert from proxy type to target type.
    pub(crate) shape_level_proxy: Option<&'static facet_core::ProxyDef>,

    /// Index of the precomputed TypePlan node for this frame's type.
    /// This is navigated in parallel with the value - when we begin_nth_field,
    /// the new frame gets the index for that field's child plan node.
    /// Use `plan.node(type_plan)` to get the actual `&TypePlanNode`.
    /// Always present - TypePlan is built for what we actually deserialize into
    /// (including proxies).
    pub(crate) type_plan: typeplan::NodeId,
}

/// A fully initialized smart-pointer staging allocation held until a deferred
/// parent frame can consume it.
///
/// The staging shape is intentionally retained separately from the pointer
/// pointee. For example, `Cow<str>` stages a sized `String`, not a wide `str`.
#[derive(Debug)]
pub(crate) struct PendingSmartPointerInner {
    data: PtrUninit,
    allocated: AllocatedShape,
    ownership: FrameOwnership,
}

#[derive(Debug)]
pub(crate) enum Tracker {
    /// Simple scalar value - no partial initialization tracking needed.
    /// Whether it's initialized is tracked by `Frame::is_init`.
    Scalar,

    /// Partially initialized array
    Array {
        /// Track which array elements are initialized (up to 63 elements)
        iset: ISet,
        /// If we're pushing another frame, this is set to the array index
        current_child: Option<usize>,
    },

    /// Partially initialized struct/tuple-struct etc.
    Struct {
        /// fields need to be individually tracked — we only
        /// support up to 63 fields.
        iset: ISet,
        /// if we're pushing another frame, this is set to the index of the struct field
        current_child: Option<usize>,
    },

    /// Smart pointer being initialized.
    /// Whether it's initialized is tracked by `Frame::is_init`.
    SmartPointer {
        /// Whether we're currently building the inner value
        building_inner: bool,
        /// Pending inner value allocation to be finalized into the smart pointer.
        /// Deferred processing requires keeping the inner value's memory stable,
        /// including its actual staging shape, allocation size, and cleanup
        /// authority, until the SmartPointer frame is finalized.
        /// None = no pending inner, Some = inner value ready for finalization.
        pending_inner: Option<PendingSmartPointerInner>,
    },

    /// We're initializing an `Arc<[T]>`, `Box<[T]>`, `Rc<[T]>`, etc.
    ///
    /// We're using the slice builder API to construct the slice
    SmartPointerSlice {
        /// The slice builder vtable
        vtable: &'static SliceBuilderVTable,

        /// Whether we're currently building an item to push
        building_item: bool,

        /// Current element index being built (for path derivation in deferred mode)
        current_child: Option<usize>,
    },

    /// Transparent inner type wrapper (`NonZero<T>`, ByteString, etc.)
    /// Used to distinguish inner frames from their parent for deferred path tracking.
    Inner {
        /// Whether we're currently building the inner value
        building_inner: bool,
    },

    /// Partially initialized enum (but we picked a variant,
    /// so it's not Uninit)
    Enum {
        /// Variant chosen for the enum
        variant: &'static Variant,
        /// Index of the variant in the enum's variants array
        variant_idx: usize,
        /// tracks enum fields (for the given variant)
        data: ISet,
        /// If we're pushing another frame, this is set to the field index
        current_child: Option<usize>,
    },

    /// Partially initialized list (Vec, etc.)
    /// Whether it's initialized is tracked by `Frame::is_init`.
    List {
        /// If we're pushing another frame for an element, this is the element index
        current_child: Option<usize>,
        /// Stable rope storage for elements during list building.
        /// A rope is a list of fixed-size chunks - chunks never reallocate, only new
        /// chunks are added. This keeps element pointers stable, enabling deferred
        /// frame processing for nested structs inside Vec elements.
        /// On finalization, elements are moved into the real Vec.
        rope: Option<ListRope>,
    },

    /// Partially initialized map (HashMap, BTreeMap, etc.)
    /// Whether it's initialized is tracked by `Frame::is_init`.
    Map {
        /// State of the current insertion operation
        insert_state: MapInsertState,
        /// Pending key-value entries to be inserted on map finalization.
        /// Deferred processing requires keeping buffers alive until finish_deferred(),
        /// so we delay actual insertion until the map frame is finalized.
        /// Each entry is (key_ptr, value_ptr) - both are initialized and owned by this tracker.
        pending_entries: Vec<(PtrUninit, Option<PtrUninit>)>,
        /// The current entry index, used for building unique paths for deferred frame storage.
        /// Incremented each time we start a new key (in begin_key).
        /// This allows inner frames of different map entries to have distinct paths.
        current_entry_index: Option<usize>,
        /// Whether we're currently building a key (true) or value (false).
        /// Used to determine whether to push MapKey or MapValue to the path.
        building_key: bool,
    },

    /// Partially initialized set (HashSet, BTreeSet, etc.)
    /// Whether it's initialized is tracked by `Frame::is_init`.
    Set {
        /// If we're pushing another frame for an element
        current_child: bool,
    },

    /// Option being initialized with Some(inner_value)
    Option {
        /// Whether we're currently building the inner value
        building_inner: bool,
        /// Pending inner value pointer to be moved with init_some on finalization.
        /// Deferred processing requires keeping the inner value's memory stable,
        /// so we delay the init_some() call until the Option frame is finalized.
        /// None = no pending inner, Some = inner value ready to be moved into Option.
        pending_inner: Option<PtrUninit>,
    },

    /// Result being initialized with Ok or Err
    Result {
        /// Whether we're building Ok (true) or Err (false)
        is_ok: bool,
        /// Whether we're currently building the inner value
        building_inner: bool,
    },

    /// Dynamic value (e.g., facet_value::Value) being initialized
    DynamicValue {
        /// What kind of dynamic value we're building
        state: DynamicValueState,
    },
}

/// State for building a dynamic value
#[derive(Debug)]
#[allow(dead_code)] // Some variants are for future use (object support)
pub(crate) enum DynamicValueState {
    /// Not yet initialized - will be set to scalar, array, or object
    Uninit,
    /// Initialized as a scalar (null, bool, number, string, bytes)
    Scalar,
    /// Initialized as an array, currently building an element
    Array {
        building_element: bool,
        /// Pending elements to be inserted during finalization (deferred mode)
        pending_elements: alloc::vec::Vec<PtrUninit>,
    },
    /// Initialized as an object
    Object {
        insert_state: DynamicObjectInsertState,
        /// Pending entries to be inserted during finalization (deferred mode)
        pending_entries: alloc::vec::Vec<(alloc::string::String, PtrUninit)>,
    },
}

/// State for inserting into a dynamic object
#[derive(Debug)]
#[allow(dead_code)] // For future use (object support)
pub(crate) enum DynamicObjectInsertState {
    /// Idle - ready for a new key-value pair
    Idle,
    /// Currently building the value for a key
    BuildingValue {
        /// The key for the current entry
        key: alloc::string::String,
    },
}

impl Tracker {
    const fn kind(&self) -> TrackerKind {
        match self {
            Tracker::Scalar => TrackerKind::Scalar,
            Tracker::Array { .. } => TrackerKind::Array,
            Tracker::Struct { .. } => TrackerKind::Struct,
            Tracker::SmartPointer { .. } => TrackerKind::SmartPointer,
            Tracker::SmartPointerSlice { .. } => TrackerKind::SmartPointerSlice,
            Tracker::Enum { .. } => TrackerKind::Enum,
            Tracker::List { .. } => TrackerKind::List,
            Tracker::Map { .. } => TrackerKind::Map,
            Tracker::Set { .. } => TrackerKind::Set,
            Tracker::Option { .. } => TrackerKind::Option,
            Tracker::Result { .. } => TrackerKind::Result,
            Tracker::DynamicValue { .. } => TrackerKind::DynamicValue,
            Tracker::Inner { .. } => TrackerKind::Inner,
        }
    }

    /// Set the current_child index for trackers that support it
    const fn set_current_child(&mut self, idx: usize) {
        match self {
            Tracker::Struct { current_child, .. }
            | Tracker::Enum { current_child, .. }
            | Tracker::Array { current_child, .. } => {
                *current_child = Some(idx);
            }
            _ => {}
        }
    }

    /// Clear the current_child index for trackers that support it
    fn clear_current_child(&mut self) {
        match self {
            Tracker::Struct { current_child, .. }
            | Tracker::Enum { current_child, .. }
            | Tracker::Array { current_child, .. }
            | Tracker::List { current_child, .. } => {
                *current_child = None;
            }
            Tracker::Set { current_child } => {
                *current_child = false;
            }
            _ => {}
        }
    }
}

impl PendingSmartPointerInner {
    /// Transfers a fully initialized child frame into deferred smart-pointer
    /// staging. Callers must validate the frame before erasing its tracker.
    fn from_initialized_frame(frame: Frame) -> Self {
        Self {
            data: frame.data,
            allocated: frame.allocated,
            ownership: frame.ownership,
        }
    }

    #[inline]
    fn shape(&self) -> &'static Shape {
        self.allocated.shape()
    }

    #[inline]
    fn data(&self) -> PtrUninit {
        self.data
    }

    /// Dispose of a staging value which remains initialized.
    fn drop_and_dealloc(self) {
        if !matches!(self.ownership, FrameOwnership::BorrowedInPlace) {
            unsafe {
                self.allocated
                    .shape()
                    .call_drop_in_place(self.data.assume_init());
            }
        }
        self.dealloc_after_move();
    }

    /// Dispose only of the allocation after ownership of its value was moved.
    fn dealloc_after_move(self) {
        if self.ownership.needs_dealloc() {
            unsafe { self.allocated.deallocate(self.data) };
        }
    }
}

impl Frame {
    fn new(
        data: PtrUninit,
        allocated: AllocatedShape,
        ownership: FrameOwnership,
        type_plan: typeplan::NodeId,
    ) -> Self {
        // For empty structs (structs with 0 fields), start as initialized since there's nothing to initialize
        // This includes empty tuples () which are zero-sized types with no fields to initialize
        let is_init = matches!(
            allocated.shape().ty,
            Type::User(UserType::Struct(struct_type)) if struct_type.fields.is_empty()
        );

        Self {
            data,
            allocated,
            is_init,
            tracker: Tracker::Scalar,
            ownership,
            using_custom_deserialization: false,
            shape_level_proxy: None,
            type_plan,
        }
    }

    /// Releases deferred smart-pointer staging without touching the pointer storage.
    ///
    /// A pending staging value owns its own allocation even when this frame borrows
    /// its destination storage (for example, an External root supplied by a caller).
    /// It therefore must be drained independently of the frame's ownership policy.
    /// Returns whether staging was present.
    fn drop_pending_smart_pointer_staging(&mut self) -> bool {
        let Tracker::SmartPointer { pending_inner, .. } = &mut self.tracker else {
            return false;
        };

        let Some(pending_inner) = pending_inner.take() else {
            return false;
        };

        pending_inner.drop_and_dealloc();
        true
    }

    /// Deinitialize a canceled frame after its child frames have been cleaned up
    /// and their bits cleared in this frame's tracker.
    ///
    /// At this point, an enum with all its field bits set is a complete value,
    /// so its custom destructor must run too. This cannot be inferred at every
    /// deinit call site: deferred parent bits may precede child validation.
    fn deinit_for_cancellation(&mut self) {
        if matches!(
            &self.tracker,
            Tracker::Enum { variant, data, .. } if data.all_set(variant.data.fields.len())
        ) {
            self.tracker = Tracker::Scalar;
            self.is_init = true;
        }
        self.deinit();
    }

    /// Deinitialize any initialized field: calls `drop_in_place` but does not free any
    /// memory even if the frame owns that memory.
    ///
    /// After this call, `is_init` will be false and `tracker` will be [Tracker::Scalar].
    fn deinit(&mut self) {
        // For BorrowedInPlace frames, we must NOT drop. These point into existing
        // collection entries (Value objects, Option inners) where the parent has no
        // per-entry tracking. Dropping here would cause double-free when parent drops.
        if matches!(self.ownership, FrameOwnership::BorrowedInPlace) {
            self.drop_pending_smart_pointer_staging();
            self.is_init = false;
            self.tracker = Tracker::Scalar;
            return;
        }

        // For RopeSlot frames, `frame.data` points into a ListRope chunk owned by
        // the parent List's tracker. We DO need to drop the element's contents
        // (respecting partial init), because with consume-time rope marking, if a
        // RopeSlot frame is still alive at deinit time the rope's initialized_count
        // does NOT cover this slot — so `ListRope::drain_into` won't drop it.
        // The drop below runs `call_drop_in_place` against `self.data` (in-rope);
        // `dealloc` is skipped since `needs_dealloc()` is false for RopeSlot, so
        // the chunk allocation stays intact for the parent rope to reclaim.

        // Field frames are responsible for their value during cleanup.
        // The ownership model ensures no double-free:
        // - begin_field: parent's iset[idx] is cleared (parent relinquishes responsibility)
        // - end: parent's iset[idx] is set (parent reclaims responsibility), frame is popped
        // So if Field frame is still on stack during cleanup, parent's iset[idx] is false,
        // meaning the parent won't drop this field - the Field frame must do it.

        match &mut self.tracker {
            Tracker::Scalar => {
                // Simple scalar - drop if initialized
                if self.is_init {
                    unsafe {
                        self.allocated
                            .shape()
                            .call_drop_in_place(self.data.assume_init())
                    };
                }
            }
            Tracker::Array { iset, .. } => {
                // Drop initialized array elements
                if let Type::Sequence(facet_core::SequenceType::Array(array_def)) =
                    self.allocated.shape().ty
                {
                    let element_layout = array_def.t.layout.sized_layout().ok();
                    if let Some(layout) = element_layout {
                        for idx in 0..array_def.n {
                            if iset.get(idx) {
                                let offset = layout.size() * idx;
                                let element_ptr = unsafe { self.data.field_init(offset) };
                                unsafe { array_def.t.call_drop_in_place(element_ptr) };
                            }
                        }
                    }
                }
            }
            Tracker::Struct { iset, .. } => {
                // Drop initialized struct fields
                if let Type::User(UserType::Struct(struct_type)) = self.allocated.shape().ty {
                    if iset.all_set(struct_type.fields.len()) {
                        unsafe {
                            self.allocated
                                .shape()
                                .call_drop_in_place(self.data.assume_init())
                        };
                    } else {
                        for (idx, field) in struct_type.fields.iter().enumerate() {
                            if iset.get(idx) {
                                // This field was initialized, drop it
                                let field_ptr = unsafe { self.data.field_init(field.offset) };
                                unsafe { field.shape().call_drop_in_place(field_ptr) };
                            }
                        }
                    }
                }
            }
            Tracker::Enum { variant, data, .. } => {
                // Drop initialized enum variant fields
                for (idx, field) in variant.data.fields.iter().enumerate() {
                    if data.get(idx) {
                        // This field was initialized, drop it
                        let field_ptr = unsafe { self.data.field_init(field.offset) };
                        unsafe { field.shape().call_drop_in_place(field_ptr) };
                    }
                }
            }
            Tracker::SmartPointer { pending_inner, .. } => {
                // A pending staging allocation is the sole owner of its
                // cleanup. The pointer storage itself remains uninitialized
                // until finalization succeeds.
                let had_pending = pending_inner.is_some();
                if let Some(pending_inner) = pending_inner.take() {
                    pending_inner.drop_and_dealloc();
                }
                // Drop an actual initialized SmartPointer, but never attempt
                // to drop its uninitialized storage while staging is pending.
                if self.is_init && !had_pending {
                    unsafe {
                        self.allocated
                            .shape()
                            .call_drop_in_place(self.data.assume_init())
                    };
                }
            }
            Tracker::SmartPointerSlice { vtable, .. } => {
                // Free the slice builder
                let builder_ptr = unsafe { self.data.assume_init() };
                unsafe {
                    (vtable.free_fn)(builder_ptr);
                }
            }
            Tracker::List { rope, .. } => {
                // Drain any rope elements first. `is_init` only indicates that the Vec
                // has been allocated (via `init_in_place_with_capacity`); elements pushed
                // via `begin_list_item` live in the rope until `drain_rope_into_vec` moves
                // them into the Vec. A successful drain leaves `rope = None` (via `.take()`),
                // so if we see `rope = Some(..)` here the elements inside were never moved
                // into the Vec and they're still owned by the rope. Drop them now.
                if let Some(mut rope) = rope.take()
                    && let Def::List(list_def) = self.allocated.shape().def
                {
                    let element_shape = list_def.t;
                    unsafe {
                        rope.drain_into(|ptr| {
                            element_shape.call_drop_in_place(PtrMut::new(ptr.as_ptr()));
                        });
                    }
                }

                // Now drop the Vec (and whatever elements it already owns).
                if self.is_init {
                    unsafe {
                        self.allocated
                            .shape()
                            .call_drop_in_place(self.data.assume_init())
                    };
                }
            }
            Tracker::Map {
                pending_entries, ..
            } => {
                // Drop the initialized Map
                if self.is_init {
                    unsafe {
                        self.allocated
                            .shape()
                            .call_drop_in_place(self.data.assume_init())
                    };
                }

                // Clean up pending entries. Each entry is `(key_ptr, Option<value_ptr>)`:
                // a full entry has `Some(value_ptr)`; a half-entry (key pushed but no value
                // paired yet) has `None`. `pending_entries` is the sole drop-and-dealloc
                // authority for any buffer it holds; entries only ever reference fully-
                // initialized buffers (pushed at walk consume-time or non-stored end()
                // paths, after validation), so drop-in-place is safe.
                if let Def::Map(map_def) = self.allocated.shape().def {
                    for (key_ptr, value_ptr) in pending_entries.drain(..) {
                        // Drop and deallocate key
                        unsafe { map_def.k().call_drop_in_place(key_ptr.assume_init()) };
                        if let Ok(key_layout) = map_def.k().layout.sized_layout()
                            && key_layout.size() > 0
                        {
                            unsafe { alloc::alloc::dealloc(key_ptr.as_mut_byte_ptr(), key_layout) };
                        }
                        // Drop and deallocate value if present (half-entries have None).
                        if let Some(value_ptr) = value_ptr {
                            unsafe { map_def.v().call_drop_in_place(value_ptr.assume_init()) };
                            if let Ok(value_layout) = map_def.v().layout.sized_layout()
                                && value_layout.size() > 0
                            {
                                unsafe {
                                    alloc::alloc::dealloc(value_ptr.as_mut_byte_ptr(), value_layout)
                                };
                            }
                        }
                    }
                }
                // Note: insert_state is no longer a cleanup source. Any key/value buffer
                // for an in-flight frame is still owned by that frame (the stored frame's
                // own dealloc handles it during `cleanup_stored_frames_on_error`, or the
                // on-stack frame's dealloc handles it in `Drop::drop`).
            }
            Tracker::Set { .. } => {
                // Drop the initialized Set
                if self.is_init {
                    unsafe {
                        self.allocated
                            .shape()
                            .call_drop_in_place(self.data.assume_init())
                    };
                }
            }
            Tracker::Option {
                building_inner,
                pending_inner,
            } => {
                // Clean up pending inner value if it was never finalized
                let had_pending = pending_inner.is_some();
                if let Some(inner_ptr) = pending_inner.take()
                    && let Def::Option(option_def) = self.allocated.shape().def
                {
                    // Drop the inner value
                    unsafe { option_def.t.call_drop_in_place(inner_ptr.assume_init()) };
                    // Deallocate the inner buffer
                    if let Ok(layout) = option_def.t.layout.sized_layout()
                        && layout.size() > 0
                    {
                        unsafe { alloc::alloc::dealloc(inner_ptr.as_mut_byte_ptr(), layout) };
                    }
                }
                // If we're building the inner value, it will be handled by the Option vtable
                // No special cleanup needed here as the Option will either be properly
                // initialized or remain uninitialized
                if !*building_inner && !had_pending {
                    // Option is fully initialized (no pending), drop it normally
                    unsafe {
                        self.allocated
                            .shape()
                            .call_drop_in_place(self.data.assume_init())
                    };
                }
            }
            Tracker::Result { building_inner, .. } => {
                // If we're building the inner value, it will be handled by the Result vtable
                // No special cleanup needed here as the Result will either be properly
                // initialized or remain uninitialized
                if !*building_inner {
                    // Result is fully initialized, drop it normally
                    unsafe {
                        self.allocated
                            .shape()
                            .call_drop_in_place(self.data.assume_init())
                    };
                }
            }
            Tracker::DynamicValue { state } => {
                // Clean up pending_entries if this is an Object
                if let DynamicValueState::Object {
                    pending_entries, ..
                } = state
                {
                    // Drop and deallocate any pending values that weren't inserted
                    if let Def::DynamicValue(dyn_def) = self.allocated.shape().def {
                        let value_shape = self.allocated.shape(); // Value entries are same shape
                        for (_key, value_ptr) in pending_entries.drain(..) {
                            // Drop the value
                            unsafe {
                                value_shape.call_drop_in_place(value_ptr.assume_init());
                            }
                            // Deallocate the value buffer
                            if let Ok(layout) = value_shape.layout.sized_layout()
                                && layout.size() > 0
                            {
                                unsafe {
                                    alloc::alloc::dealloc(value_ptr.as_mut_byte_ptr(), layout);
                                }
                            }
                        }
                        // Note: keys are Strings and will be dropped when pending_entries is dropped
                        let _ = dyn_def; // silence unused warning
                    }
                }

                // Clean up pending_elements if this is an Array
                if let DynamicValueState::Array {
                    pending_elements, ..
                } = state
                {
                    // Drop and deallocate any pending elements that weren't inserted
                    let element_shape = self.allocated.shape(); // Array elements are same shape
                    for element_ptr in pending_elements.drain(..) {
                        // Drop the element
                        unsafe {
                            element_shape.call_drop_in_place(element_ptr.assume_init());
                        }
                        // Deallocate the element buffer
                        if let Ok(layout) = element_shape.layout.sized_layout()
                            && layout.size() > 0
                        {
                            unsafe {
                                alloc::alloc::dealloc(element_ptr.as_mut_byte_ptr(), layout);
                            }
                        }
                    }
                }

                // Drop if initialized
                if self.is_init {
                    let result = unsafe {
                        self.allocated
                            .shape()
                            .call_drop_in_place(self.data.assume_init())
                    };
                    if result.is_none() {
                        // This would be a bug - DynamicValue should always have drop_in_place
                        panic!(
                            "DynamicValue type {} has no drop_in_place implementation",
                            self.allocated.shape()
                        );
                    }
                }
            }
            Tracker::Inner { .. } => {
                // Inner wrapper - drop if initialized
                if self.is_init {
                    unsafe {
                        self.allocated
                            .shape()
                            .call_drop_in_place(self.data.assume_init())
                    };
                }
            }
        }

        self.is_init = false;
        self.tracker = Tracker::Scalar;
    }

    /// Deinitialize any initialized value for REPLACEMENT purposes.
    ///
    /// Unlike `deinit()` which is used during error cleanup, this method is used when
    /// we're about to overwrite a value with a new one (e.g., in `set_shape`).
    ///
    /// The difference is important for Field frames with simple trackers:
    /// - During cleanup: parent struct will drop all initialized fields, so Field frames skip dropping
    /// - During replacement: we're about to overwrite, so we MUST drop the old value
    ///
    /// For BorrowedInPlace frames: same logic applies - we must drop when replacing.
    fn deinit_for_replace(&mut self) {
        // For BorrowedInPlace frames, deinit() skips dropping (parent owns on cleanup).
        // But when REPLACING a value, we must drop the old value first.
        if matches!(self.ownership, FrameOwnership::BorrowedInPlace) && self.is_init {
            unsafe {
                self.allocated
                    .shape()
                    .call_drop_in_place(self.data.assume_init());
            }

            // CRITICAL: For DynamicValue (e.g., facet_value::Value), the parent Object's
            // HashMap entry still points to this location. If we just drop and leave garbage,
            // the parent will try to drop that garbage when it's cleaned up, causing
            // use-after-free. We must reinitialize to a safe default (Null) so the parent
            // can safely drop it later.
            if let Def::DynamicValue(dyn_def) = &self.allocated.shape().def {
                unsafe {
                    (dyn_def.vtable.set_null)(self.data);
                }
                // Keep is_init = true since we just initialized it to Null
                self.tracker = Tracker::DynamicValue {
                    state: DynamicValueState::Scalar,
                };
                return;
            }

            self.is_init = false;
            self.tracker = Tracker::Scalar;
            return;
        }

        // Field frames handle their own cleanup in deinit() - no special handling needed here.

        // All other cases: use normal deinit
        self.deinit();
    }

    /// This must be called after (fully) initializing a value.
    ///
    /// This sets `is_init` to `true` to indicate the value is initialized.
    /// Composite types (structs, enums, etc.) might be handled differently.
    ///
    /// # Safety
    ///
    /// This should only be called when `self.data` has been actually initialized.
    const unsafe fn mark_as_init(&mut self) {
        self.is_init = true;
    }

    /// Deallocate the memory associated with this frame, if it owns it.
    ///
    /// The memory has to be deinitialized first, see [Frame::deinit]
    fn dealloc(self) {
        // Only deallocate if this frame owns its memory
        if !self.ownership.needs_dealloc() {
            return;
        }

        // If we need to deallocate, the frame must be deinitialized first
        if self.is_init {
            unreachable!("a frame has to be deinitialized before being deallocated")
        }

        unsafe { self.allocated.deallocate(self.data) };
    }

    /// Fill in defaults for any unset fields that have default values.
    ///
    /// This handles:
    /// - Container-level defaults (when no fields set and struct has Default impl)
    /// - Fields with `#[facet(default = ...)]` - uses the explicit default function
    /// - Fields with `#[facet(default)]` - uses the type's Default impl
    /// - `Option<T>` fields - default to None
    ///
    /// Returns Ok(()) if successful, or an error if a field has `#[facet(default)]`
    /// but no default implementation is available.
    fn fill_defaults(&mut self) -> Result<(), ReflectErrorKind> {
        // First, check if we need to upgrade from Scalar to Struct tracker
        // This happens when no fields were visited at all in deferred mode
        if !self.is_init
            && matches!(self.tracker, Tracker::Scalar)
            && let Type::User(UserType::Struct(struct_type)) = self.allocated.shape().ty
        {
            // If no fields were visited and the container has a default, use it
            // SAFETY: We're about to initialize the entire struct with its default value
            if unsafe { self.allocated.shape().call_default_in_place(self.data) }.is_some() {
                self.is_init = true;
                return Ok(());
            }
            // Otherwise initialize the struct tracker with empty iset
            self.tracker = Tracker::Struct {
                iset: ISet::new(struct_type.fields.len()),
                current_child: None,
            };
        }

        // Handle Option types with Scalar tracker - default to None
        // This happens in deferred mode when an Option field was never touched
        if !self.is_init
            && matches!(self.tracker, Tracker::Scalar)
            && matches!(self.allocated.shape().def, Def::Option(_))
        {
            // SAFETY: Option<T> always implements Default (as None)
            if unsafe { self.allocated.shape().call_default_in_place(self.data) }.is_some() {
                self.is_init = true;
                return Ok(());
            }
        }

        match &mut self.tracker {
            Tracker::Struct { iset, .. } => {
                if let Type::User(UserType::Struct(struct_type)) = self.allocated.shape().ty {
                    // Fast path: if ALL fields are set, nothing to do
                    if iset.all_set(struct_type.fields.len()) {
                        return Ok(());
                    }

                    // Check if NO fields have been set and the container has a default
                    let no_fields_set = (0..struct_type.fields.len()).all(|i| !iset.get(i));
                    if no_fields_set {
                        // SAFETY: We're about to initialize the entire struct with its default value
                        if unsafe { self.allocated.shape().call_default_in_place(self.data) }
                            .is_some()
                        {
                            self.tracker = Tracker::Scalar;
                            self.is_init = true;
                            return Ok(());
                        }
                    }

                    // Check if the container has #[facet(default)] attribute
                    let container_has_default = self.allocated.shape().has_default_attr();

                    // Fill defaults for individual fields
                    for (idx, field) in struct_type.fields.iter().enumerate() {
                        // Skip already-initialized fields
                        if iset.get(idx) {
                            continue;
                        }

                        // Calculate field pointer
                        let field_ptr = unsafe { self.data.field_uninit(field.offset) };

                        // Try to initialize with default
                        if unsafe {
                            Self::try_init_field_default(field, field_ptr, container_has_default)
                        } {
                            // Mark field as initialized
                            iset.set(idx);
                        } else if field.has_default() {
                            // Field has #[facet(default)] but we couldn't find a default function.
                            // This happens with opaque types that don't have default_in_place.
                            return Err(ReflectErrorKind::DefaultAttrButNoDefaultImpl {
                                shape: field.shape(),
                            });
                        }
                    }
                }
            }
            Tracker::Enum { variant, data, .. } => {
                // Fast path: if ALL fields are set, nothing to do
                let num_fields = variant.data.fields.len();
                if num_fields == 0 || data.all_set(num_fields) {
                    return Ok(());
                }

                // Check if the container has #[facet(default)] attribute
                let container_has_default = self.allocated.shape().has_default_attr();

                // Handle enum variant fields
                for (idx, field) in variant.data.fields.iter().enumerate() {
                    // Skip already-initialized fields
                    if data.get(idx) {
                        continue;
                    }

                    // Calculate field pointer within the variant data
                    let field_ptr = unsafe { self.data.field_uninit(field.offset) };

                    // Try to initialize with default
                    if unsafe {
                        Self::try_init_field_default(field, field_ptr, container_has_default)
                    } {
                        // Mark field as initialized
                        data.set(idx);
                    } else if field.has_default() {
                        // Field has #[facet(default)] but we couldn't find a default function.
                        return Err(ReflectErrorKind::DefaultAttrButNoDefaultImpl {
                            shape: field.shape(),
                        });
                    }
                }
            }
            // Other tracker types don't have fields with defaults
            _ => {}
        }
        Ok(())
    }

    /// Initialize a field with its default value if one is available.
    ///
    /// Priority:
    /// 1. Explicit field-level default_fn (from `#[facet(default = ...)]`)
    /// 2. Type-level default_in_place (from Default impl, including `Option<T>`)
    ///    but only if the field has the DEFAULT flag
    /// 3. Container-level default: if the container has `#[facet(default)]` and
    ///    the field's type implements Default, use that
    /// 4. Special cases: `Option<T>` (defaults to None), () (unit type)
    ///
    /// Returns true if a default was applied, false otherwise.
    ///
    /// # Safety
    ///
    /// `field_ptr` must point to uninitialized memory of the appropriate type.
    unsafe fn try_init_field_default(
        field: &Field,
        field_ptr: PtrUninit,
        container_has_default: bool,
    ) -> bool {
        use facet_core::DefaultSource;

        // First check for explicit field-level default
        if let Some(default_source) = field.default {
            match default_source {
                DefaultSource::Custom(default_fn) => {
                    // Custom default function - it expects PtrUninit
                    unsafe { default_fn(field_ptr) };
                    return true;
                }
                DefaultSource::FromTrait => {
                    // Use the type's Default trait
                    if unsafe { field.shape().call_default_in_place(field_ptr) }.is_some() {
                        return true;
                    }
                }
                _ => {
                    // Unknown default sources fall back to the type's Default trait.
                    if unsafe { field.shape().call_default_in_place(field_ptr) }.is_some() {
                        return true;
                    }
                }
            }
        }

        // If container has #[facet(default)] and the field's type implements Default,
        // use the type's Default impl. This allows `#[facet(default)]` on a struct to
        // mean "use Default for any missing fields whose types implement Default".
        if container_has_default
            && unsafe { field.shape().call_default_in_place(field_ptr) }.is_some()
        {
            return true;
        }

        // Special case: Option<T> always defaults to None, even without explicit #[facet(default)]
        // This is because Option is fundamentally "optional" - if not set, it should be None
        if matches!(field.shape().def, Def::Option(_))
            && unsafe { field.shape().call_default_in_place(field_ptr) }.is_some()
        {
            return true;
        }

        // Special case: () unit type always defaults to ()
        if field.shape().is_type::<()>()
            && unsafe { field.shape().call_default_in_place(field_ptr) }.is_some()
        {
            return true;
        }

        // Special case: Collection types (Vec, HashMap, HashSet, etc.) default to empty
        // These types have obvious "zero values" and it's almost always what you want
        // when deserializing data where the collection is simply absent.
        if matches!(field.shape().def, Def::List(_) | Def::Map(_) | Def::Set(_))
            && unsafe { field.shape().call_default_in_place(field_ptr) }.is_some()
        {
            return true;
        }

        false
    }

    /// Drain all initialized elements from the rope into the Vec.
    ///
    /// This is called when finalizing a list that used rope storage. Elements were
    /// built in stable rope chunks to allow deferred processing; now we move them
    /// into the actual Vec.
    ///
    /// # Safety
    ///
    /// The rope must contain only initialized elements (via `mark_last_initialized`).
    /// The list_data must point to an initialized Vec with capacity for the elements.
    fn drain_rope_into_vec(
        mut rope: ListRope,
        list_def: &facet_core::ListDef,
        list_data: PtrUninit,
    ) -> Result<(), ReflectErrorKind> {
        let count = rope.initialized_count();
        if count == 0 {
            return Ok(());
        }

        let push_fn = list_def
            .push()
            .ok_or_else(|| ReflectErrorKind::OperationFailed {
                shape: list_def.t(),
                operation: "List missing push function for rope drain",
            })?;

        // SAFETY: list_data points to initialized Vec (is_init was true)
        let list_ptr = unsafe { list_data.assume_init() };

        // Reserve space if available (optimization, not required)
        if let Some(reserve_fn) = list_def.reserve() {
            unsafe {
                reserve_fn(list_ptr, count);
            }
        }

        // Move each element from rope to Vec
        // SAFETY: rope contains `count` initialized elements
        unsafe {
            rope.drain_into(|element_ptr| {
                push_fn(
                    facet_core::PtrMut::new(list_ptr.as_mut_byte_ptr()),
                    facet_core::PtrMut::new(element_ptr.as_ptr()),
                );
            });
        }

        Ok(())
    }

    /// Insert all pending key-value entries into the map.
    ///
    /// This is called when finalizing a map that used delayed insertion. Entries were
    /// kept in pending_entries to allow deferred processing; now we insert them into
    /// the actual map and deallocate the temporary buffers.
    fn drain_pending_into_map(
        pending_entries: &mut Vec<(PtrUninit, Option<PtrUninit>)>,
        map_def: &facet_core::MapDef,
        map_data: PtrUninit,
    ) -> Result<(), ReflectErrorKind> {
        let insert_fn = map_def.vtable.insert;

        // SAFETY: map_data points to initialized map (is_init was true)
        let map_ptr = unsafe { map_data.assume_init() };

        for (key_ptr, value_ptr) in pending_entries.drain(..) {
            // Every entry at finalize time MUST be a full (key, value) pair. A half-entry
            // (value is None) means a key was pushed without a paired value — that's an
            // invariant violation.
            let Some(value_ptr) = value_ptr else {
                return Err(ReflectErrorKind::InvariantViolation {
                    invariant: "map pending_entries contains half-entry (key without value) at finalize",
                });
            };
            // Insert the key-value pair
            unsafe {
                insert_fn(
                    facet_core::PtrMut::new(map_ptr.as_mut_byte_ptr()),
                    facet_core::PtrMut::new(key_ptr.as_mut_byte_ptr()),
                    facet_core::PtrMut::new(value_ptr.as_mut_byte_ptr()),
                );
            }

            // Deallocate the temporary buffers (insert moved the data)
            if let Ok(key_layout) = map_def.k().layout.sized_layout()
                && key_layout.size() > 0
            {
                unsafe { alloc::alloc::dealloc(key_ptr.as_mut_byte_ptr(), key_layout) };
            }
            if let Ok(value_layout) = map_def.v().layout.sized_layout()
                && value_layout.size() > 0
            {
                unsafe { alloc::alloc::dealloc(value_ptr.as_mut_byte_ptr(), value_layout) };
            }
        }

        Ok(())
    }

    /// Complete an Option by moving the pending inner value into it.
    ///
    /// This is called when finalizing an Option that used deferred init_some.
    /// The inner value was kept in stable memory for deferred processing;
    /// now we move it into the Option and deallocate the temporary buffer.
    fn complete_pending_option(
        option_def: facet_core::OptionDef,
        option_data: PtrUninit,
        inner_ptr: PtrUninit,
    ) -> Result<(), ReflectErrorKind> {
        let init_some_fn = option_def.vtable.init_some;
        let inner_shape = option_def.t;

        // The inner_ptr contains the initialized inner value
        let inner_value_ptr = unsafe { inner_ptr.assume_init() };

        // Initialize the Option as Some(inner_value)
        unsafe {
            init_some_fn(option_data, inner_value_ptr);
        }

        // Deallocate the inner value's memory since init_some_fn moved it
        if let Ok(layout) = inner_shape.layout.sized_layout()
            && layout.size() > 0
        {
            unsafe { alloc::alloc::dealloc(inner_ptr.as_mut_byte_ptr(), layout) };
        }

        Ok(())
    }

    /// Construct an independent pointer through its explicit borrow-and-promote
    /// capabilities.
    ///
    /// This is deliberately limited to matching, sized pointees. In particular,
    /// the `str` builder stores a `String` in the temporary buffer, so it must use
    /// its dedicated conversion below rather than pass a thin `String` pointer to
    /// a hook expecting a wide `str` pointer.
    ///
    /// Returns `true` when both hooks were present and the destination was
    /// initialized. The caller still owns and must dispose of `inner_data`.
    fn try_borrow_and_promote_smart_pointer(
        smart_ptr_def: facet_core::PointerDef,
        smart_ptr_data: PtrUninit,
        inner_shape: &'static Shape,
        inner_data: PtrUninit,
    ) -> bool {
        let (Some(borrow_from_pointee_fn), Some(promote_to_owned_fn), Some(pointee_shape)) = (
            smart_ptr_def.vtable.borrow_from_pointee_fn,
            smart_ptr_def.vtable.promote_to_owned_fn,
            smart_ptr_def.pointee,
        ) else {
            return false;
        };

        if !pointee_shape.is_shape(inner_shape) || pointee_shape.layout.sized_layout().is_err() {
            return false;
        }

        // SAFETY: callers pass initialized storage for the matching, sized pointee.
        let inner_data = unsafe { inner_data.assume_init() };
        // SAFETY: the paired hooks' contracts make the destination initialized and
        // independent before this function returns. The source remains initialized,
        // so the caller must still drop it.
        unsafe {
            borrow_from_pointee_fn(smart_ptr_data, inner_data.as_const());
            promote_to_owned_fn(smart_ptr_data.assume_init());
        }
        true
    }

    fn complete_pending_smart_pointer(
        smart_ptr_shape: &'static Shape,
        smart_ptr_def: facet_core::PointerDef,
        smart_ptr_data: PtrUninit,
        pending_inner: PendingSmartPointerInner,
    ) -> Result<(), ReflectErrorKind> {
        let inner_shape = pending_inner.shape();
        let inner_data = pending_inner.data();

        // Check the owned-transfer construction path first. It can only consume
        // a staging allocation whose actual shape matches the reflected pointee.
        if let (Some(new_into_fn), Some(pointee_shape)) =
            (smart_ptr_def.vtable.new_into_fn, smart_ptr_def.pointee)
            && pointee_shape.is_shape(inner_shape)
        {
            unsafe { new_into_fn(smart_ptr_data, inner_data.assume_init()) };
            pending_inner.dealloc_after_move();
            return Ok(());
        }

        // Some pointer types (currently `Cow`) first borrow a matching pointee and
        // then materialize themselves before the source is released. Unlike
        // `new_into_fn`, borrowing does not consume the source, so drop it before
        // releasing the temporary allocation.
        if Self::try_borrow_and_promote_smart_pointer(
            smart_ptr_def,
            smart_ptr_data,
            inner_shape,
            inner_data,
        ) {
            pending_inner.drop_and_dealloc();
            return Ok(());
        }

        // Check for unsized pointee case: String -> Arc<str>/Box<str>/Rc<str>
        if let Some(pointee) = smart_ptr_def.pointee()
            && pointee.is_shape(str::SHAPE)
            && inner_shape.is_shape(alloc::string::String::SHAPE)
        {
            use alloc::{borrow::Cow, rc::Rc, string::String, sync::Arc};
            use facet_core::KnownPointer;

            let Some(known) = smart_ptr_def.known else {
                pending_inner.drop_and_dealloc();
                return Err(ReflectErrorKind::OperationFailed {
                    shape: smart_ptr_shape,
                    operation: "SmartPointer<str> missing known pointer type",
                });
            };

            if !matches!(
                known,
                KnownPointer::Box | KnownPointer::Arc | KnownPointer::Rc | KnownPointer::Cow
            ) {
                pending_inner.drop_and_dealloc();
                return Err(ReflectErrorKind::OperationFailed {
                    shape: smart_ptr_shape,
                    operation: "Unsupported SmartPointer<str> type",
                });
            }

            // Read the String value from its actual staging allocation.
            let string_ptr = inner_data.as_mut_byte_ptr() as *mut String;
            let string_value = unsafe { core::ptr::read(string_ptr) };

            // Convert to the appropriate smart pointer type
            match known {
                KnownPointer::Box => {
                    let boxed: alloc::boxed::Box<str> = string_value.into_boxed_str();
                    unsafe {
                        core::ptr::write(
                            smart_ptr_data.as_mut_byte_ptr() as *mut alloc::boxed::Box<str>,
                            boxed,
                        );
                    }
                }
                KnownPointer::Arc => {
                    let arc: Arc<str> = Arc::from(string_value.into_boxed_str());
                    unsafe {
                        core::ptr::write(smart_ptr_data.as_mut_byte_ptr() as *mut Arc<str>, arc);
                    }
                }
                KnownPointer::Rc => {
                    let rc: Rc<str> = Rc::from(string_value.into_boxed_str());
                    unsafe {
                        core::ptr::write(smart_ptr_data.as_mut_byte_ptr() as *mut Rc<str>, rc);
                    }
                }
                KnownPointer::Cow => {
                    let cow: Cow<'static, str> = Cow::Owned(string_value);
                    unsafe {
                        core::ptr::write(
                            smart_ptr_data.as_mut_byte_ptr() as *mut Cow<'static, str>,
                            cow,
                        );
                    }
                }
                _ => unreachable!("supported SmartPointer<str> kind was checked above"),
            }

            // `ptr::read` moved the String, so only its exact staging allocation
            // remains to be reclaimed.
            pending_inner.dealloc_after_move();
            return Ok(());
        }

        pending_inner.drop_and_dealloc();
        Err(ReflectErrorKind::OperationFailed {
            shape: smart_ptr_shape,
            operation: "SmartPointer missing an owned construction capability and not a supported unsized type",
        })
    }

    /// Returns an error if the value is not fully initialized.
    /// For lists with rope storage, drains the rope into the Vec.
    /// For maps with pending entries, drains the entries into the map.
    /// For options with pending inner values, calls init_some.
    fn require_full_initialization(&mut self) -> Result<(), ReflectErrorKind> {
        match &mut self.tracker {
            Tracker::Scalar => {
                if self.is_init {
                    Ok(())
                } else {
                    Err(ReflectErrorKind::UninitializedValue {
                        shape: self.allocated.shape(),
                    })
                }
            }
            Tracker::Array { iset, .. } => {
                match self.allocated.shape().ty {
                    Type::Sequence(facet_core::SequenceType::Array(array_def)) => {
                        // Check if all array elements are initialized
                        if (0..array_def.n).all(|idx| iset.get(idx)) {
                            Ok(())
                        } else {
                            Err(ReflectErrorKind::UninitializedValue {
                                shape: self.allocated.shape(),
                            })
                        }
                    }
                    _ => Err(ReflectErrorKind::UninitializedValue {
                        shape: self.allocated.shape(),
                    }),
                }
            }
            Tracker::Struct { iset, .. } => {
                match self.allocated.shape().ty {
                    Type::User(UserType::Struct(struct_type)) => {
                        if iset.all_set(struct_type.fields.len()) {
                            Ok(())
                        } else {
                            // Find index of the first bit not set
                            let first_missing_idx =
                                (0..struct_type.fields.len()).find(|&idx| !iset.get(idx));
                            if let Some(missing_idx) = first_missing_idx {
                                let field_name = struct_type.fields[missing_idx].name;
                                Err(ReflectErrorKind::UninitializedField {
                                    shape: self.allocated.shape(),
                                    field_name,
                                })
                            } else {
                                // fallback, something went wrong
                                Err(ReflectErrorKind::UninitializedValue {
                                    shape: self.allocated.shape(),
                                })
                            }
                        }
                    }
                    _ => Err(ReflectErrorKind::UninitializedValue {
                        shape: self.allocated.shape(),
                    }),
                }
            }
            Tracker::Enum { variant, data, .. } => {
                // Check if all fields of the variant are initialized
                let num_fields = variant.data.fields.len();
                if num_fields == 0 {
                    // Unit variant, always initialized
                    Ok(())
                } else if (0..num_fields).all(|idx| data.get(idx)) {
                    Ok(())
                } else {
                    // Find the first uninitialized field
                    let first_missing_idx = (0..num_fields).find(|&idx| !data.get(idx));
                    if let Some(missing_idx) = first_missing_idx {
                        let field_name = variant.data.fields[missing_idx].name;
                        Err(ReflectErrorKind::UninitializedField {
                            shape: self.allocated.shape(),
                            field_name,
                        })
                    } else {
                        Err(ReflectErrorKind::UninitializedValue {
                            shape: self.allocated.shape(),
                        })
                    }
                }
            }
            Tracker::SmartPointer {
                building_inner,
                pending_inner,
            } => {
                if *building_inner {
                    // Inner value is still being built
                    Err(ReflectErrorKind::UninitializedValue {
                        shape: self.allocated.shape(),
                    })
                } else if let Some(pending_inner) = pending_inner.take() {
                    // Finalize the pending inner value
                    let smart_ptr_shape = self.allocated.shape();
                    if let Def::Pointer(smart_ptr_def) = smart_ptr_shape.def {
                        let result = Self::complete_pending_smart_pointer(
                            smart_ptr_shape,
                            smart_ptr_def,
                            self.data,
                            pending_inner,
                        );
                        self.is_init = result.is_ok();
                        result
                    } else {
                        pending_inner.drop_and_dealloc();
                        self.is_init = false;
                        Err(ReflectErrorKind::OperationFailed {
                            shape: smart_ptr_shape,
                            operation: "SmartPointer frame without SmartPointer definition",
                        })
                    }
                } else if self.is_init {
                    Ok(())
                } else {
                    Err(ReflectErrorKind::UninitializedValue {
                        shape: self.allocated.shape(),
                    })
                }
            }
            Tracker::SmartPointerSlice { building_item, .. } => {
                if *building_item {
                    Err(ReflectErrorKind::UninitializedValue {
                        shape: self.allocated.shape(),
                    })
                } else {
                    Ok(())
                }
            }
            Tracker::List {
                current_child,
                rope,
            } => {
                if self.is_init && current_child.is_none() {
                    // Drain rope into Vec if we have elements stored there
                    if let Some(rope) = rope.take()
                        && let Def::List(list_def) = self.allocated.shape().def
                    {
                        Self::drain_rope_into_vec(rope, &list_def, self.data)?;
                    }
                    Ok(())
                } else {
                    Err(ReflectErrorKind::UninitializedValue {
                        shape: self.allocated.shape(),
                    })
                }
            }
            Tracker::Map {
                insert_state,
                pending_entries,
                ..
            } => {
                if self.is_init && matches!(insert_state, MapInsertState::Idle) {
                    // Insert all pending entries into the map
                    if !pending_entries.is_empty()
                        && let Def::Map(map_def) = self.allocated.shape().def
                    {
                        Self::drain_pending_into_map(pending_entries, &map_def, self.data)?;
                    }
                    Ok(())
                } else {
                    Err(ReflectErrorKind::UninitializedValue {
                        shape: self.allocated.shape(),
                    })
                }
            }
            Tracker::Set { current_child } => {
                if self.is_init && !*current_child {
                    Ok(())
                } else {
                    Err(ReflectErrorKind::UninitializedValue {
                        shape: self.allocated.shape(),
                    })
                }
            }
            Tracker::Option {
                building_inner,
                pending_inner,
            } => {
                if *building_inner {
                    Err(ReflectErrorKind::UninitializedValue {
                        shape: self.allocated.shape(),
                    })
                } else {
                    // Finalize pending init_some if we have a pending inner value
                    if let Some(inner_ptr) = pending_inner.take()
                        && let Def::Option(option_def) = self.allocated.shape().def
                    {
                        Self::complete_pending_option(option_def, self.data, inner_ptr)?;
                    }
                    Ok(())
                }
            }
            Tracker::Result { building_inner, .. } => {
                if *building_inner {
                    Err(ReflectErrorKind::UninitializedValue {
                        shape: self.allocated.shape(),
                    })
                } else {
                    Ok(())
                }
            }
            Tracker::Inner { building_inner } => {
                if *building_inner {
                    // Inner value is still being built
                    Err(ReflectErrorKind::UninitializedValue {
                        shape: self.allocated.shape(),
                    })
                } else if self.is_init {
                    Ok(())
                } else {
                    Err(ReflectErrorKind::UninitializedValue {
                        shape: self.allocated.shape(),
                    })
                }
            }
            Tracker::DynamicValue { state } => {
                if matches!(state, DynamicValueState::Uninit) {
                    Err(ReflectErrorKind::UninitializedValue {
                        shape: self.allocated.shape(),
                    })
                } else {
                    // Insert pending entries for Object state
                    if let DynamicValueState::Object {
                        pending_entries,
                        insert_state,
                    } = state
                    {
                        if !matches!(insert_state, DynamicObjectInsertState::Idle) {
                            return Err(ReflectErrorKind::UninitializedValue {
                                shape: self.allocated.shape(),
                            });
                        }

                        if !pending_entries.is_empty()
                            && let Def::DynamicValue(dyn_def) = self.allocated.shape().def
                        {
                            let object_ptr = unsafe { self.data.assume_init() };
                            let value_shape = self.allocated.shape();

                            for (key, value_ptr) in pending_entries.drain(..) {
                                // Insert the entry
                                unsafe {
                                    (dyn_def.vtable.insert_object_entry)(
                                        object_ptr,
                                        &key,
                                        value_ptr.assume_init(),
                                    );
                                }
                                // Deallocate the value buffer (insert_object_entry moved the value)
                                if let Ok(layout) = value_shape.layout.sized_layout()
                                    && layout.size() > 0
                                {
                                    unsafe {
                                        alloc::alloc::dealloc(value_ptr.as_mut_byte_ptr(), layout);
                                    }
                                }
                            }
                        }
                    }

                    // Insert pending elements for Array state
                    if let DynamicValueState::Array {
                        pending_elements,
                        building_element,
                    } = state
                    {
                        if *building_element {
                            return Err(ReflectErrorKind::UninitializedValue {
                                shape: self.allocated.shape(),
                            });
                        }

                        if !pending_elements.is_empty()
                            && let Def::DynamicValue(dyn_def) = self.allocated.shape().def
                        {
                            let array_ptr = unsafe { self.data.assume_init() };
                            let element_shape = self.allocated.shape();

                            for element_ptr in pending_elements.drain(..) {
                                // Push the element into the array
                                unsafe {
                                    (dyn_def.vtable.push_array_element)(
                                        array_ptr,
                                        element_ptr.assume_init(),
                                    );
                                }
                                // Deallocate the element buffer (push_array_element moved the value)
                                if let Ok(layout) = element_shape.layout.sized_layout()
                                    && layout.size() > 0
                                {
                                    unsafe {
                                        alloc::alloc::dealloc(
                                            element_ptr.as_mut_byte_ptr(),
                                            layout,
                                        );
                                    }
                                }
                            }
                        }
                    }

                    Ok(())
                }
            }
        }
    }

    /// Fill defaults and check required fields in a single pass using precomputed plans.
    ///
    /// This replaces the separate `fill_defaults` + `require_full_initialization` calls
    /// with a single iteration over the precomputed `FieldInitPlan` list.
    ///
    /// # Arguments
    /// * `plans` - Precomputed field initialization plans from TypePlan
    /// * `num_fields` - Total number of fields (from StructPlan/VariantPlanMeta)
    /// * `type_plan_core` - Reference to the TypePlanCore for resolving validators
    ///
    /// # Returns
    /// `Ok(())` if all required fields are set (or filled with defaults), or an error
    /// describing the first missing required field.
    #[allow(unsafe_code)]
    fn fill_and_require_fields(
        &mut self,
        plans: &[FieldInitPlan],
        num_fields: usize,
        type_plan_core: &TypePlanCore,
    ) -> Result<(), ReflectErrorKind> {
        // With lazy tracker initialization, structs start with Tracker::Scalar.
        // If is_init is true with Scalar, the struct was set wholesale - nothing to do.
        // If is_init is false, we need to upgrade to Tracker::Struct to track fields.
        if !self.is_init
            && matches!(self.tracker, Tracker::Scalar)
            && matches!(self.allocated.shape().ty, Type::User(UserType::Struct(_)))
        {
            // Try container-level default first
            if unsafe { self.allocated.shape().call_default_in_place(self.data) }.is_some() {
                self.is_init = true;
                return Ok(());
            }
            // Upgrade to Tracker::Struct for field-by-field tracking
            self.tracker = Tracker::Struct {
                iset: ISet::new(num_fields),
                current_child: None,
            };
        }

        // Get the iset based on tracker type
        let iset = match &mut self.tracker {
            Tracker::Struct { iset, .. } => iset,
            Tracker::Enum { data, .. } => data,
            // Scalar with is_init=true means struct was set wholesale - all fields initialized
            Tracker::Scalar if self.is_init => return Ok(()),
            // Other tracker types don't use field_init_plans
            _ => return Ok(()),
        };

        // Fast path: if all fields are already set, no defaults needed.
        // But validators still need to run.
        let all_fields_set = iset.all_set(num_fields);

        for plan in plans {
            if !all_fields_set && !iset.get(plan.index) {
                // Field not set - handle according to fill rule
                match &plan.fill_rule {
                    FillRule::Defaultable(default) => {
                        // Calculate field pointer
                        let field_ptr = unsafe { self.data.field_uninit(plan.offset) };

                        // Call the appropriate default function
                        let success = match default {
                            FieldDefault::Custom(default_fn) => {
                                // SAFETY: default_fn writes to uninitialized memory
                                unsafe { default_fn(field_ptr) };
                                true
                            }
                            FieldDefault::FromTrait(shape) => {
                                // SAFETY: call_default_in_place writes to uninitialized memory
                                unsafe { shape.call_default_in_place(field_ptr) }.is_some()
                            }
                        };

                        if success {
                            iset.set(plan.index);
                        } else {
                            return Err(ReflectErrorKind::UninitializedField {
                                shape: self.allocated.shape(),
                                field_name: plan.name,
                            });
                        }
                    }
                    FillRule::Required => {
                        return Err(ReflectErrorKind::UninitializedField {
                            shape: self.allocated.shape(),
                            field_name: plan.name,
                        });
                    }
                }
            }

            // Run validators on the (now initialized) field
            if !plan.validators.is_empty() {
                let field_ptr = unsafe { self.data.field_init(plan.offset) };
                for validator in type_plan_core.validators(plan.validators) {
                    validator.run(field_ptr.into(), plan.name, self.allocated.shape())?;
                }
            }
        }

        Ok(())
    }

    /// Get the [EnumType] of the frame's shape, if it is an enum type
    pub(crate) const fn get_enum_type(&self) -> Result<EnumType, ReflectErrorKind> {
        match self.allocated.shape().ty {
            Type::User(UserType::Enum(e)) => Ok(e),
            _ => Err(ReflectErrorKind::WasNotA {
                expected: "enum",
                actual: self.allocated.shape(),
            }),
        }
    }

    pub(crate) fn get_field(&self) -> Option<&Field> {
        match self.allocated.shape().ty {
            Type::User(user_type) => match user_type {
                UserType::Struct(struct_type) => {
                    // Try to get currently active field index
                    if let Tracker::Struct {
                        current_child: Some(idx),
                        ..
                    } = &self.tracker
                    {
                        struct_type.fields.get(*idx)
                    } else {
                        None
                    }
                }
                UserType::Enum(_enum_type) => {
                    if let Tracker::Enum {
                        variant,
                        current_child: Some(idx),
                        ..
                    } = &self.tracker
                    {
                        variant.data.fields.get(*idx)
                    } else {
                        None
                    }
                }
                _ => None,
            },
            _ => None,
        }
    }
}

// Convenience methods on Partial for accessing FrameMode internals.
// These help minimize changes to the rest of the codebase during the refactor.
impl<'facet, const BORROW: bool> Partial<'facet, BORROW> {
    /// Get a reference to the frame stack.
    #[inline]
    pub(crate) const fn frames(&self) -> &Vec<Frame> {
        self.mode.stack()
    }

    /// Get a mutable reference to the frame stack.
    #[inline]
    pub(crate) fn frames_mut(&mut self) -> &mut Vec<Frame> {
        self.mode.stack_mut()
    }

    /// Check if we're in deferred mode.
    #[inline]
    pub const fn is_deferred(&self) -> bool {
        self.mode.is_deferred()
    }

    /// Get the start depth if in deferred mode.
    #[inline]
    pub(crate) const fn start_depth(&self) -> Option<usize> {
        self.mode.start_depth()
    }

    /// Derive the path from the current frame stack.
    ///
    /// Compute the navigation path for deferred mode storage and lookup.
    /// The returned `Path` is anchored to the root shape for proper type context.
    ///
    /// This extracts Field steps from struct/enum frames and Index steps from
    /// array/list frames. Option wrappers, smart pointers (Box, Rc, etc.), and
    /// other transparent types don't add path steps.
    ///
    /// This MUST match the storage path computation in end() for consistency.
    pub(crate) fn derive_path(&self) -> Path {
        // Get the root shape from the first frame
        let root_shape = self
            .frames()
            .first()
            .map(|f| f.allocated.shape())
            .unwrap_or_else(|| {
                // Fallback to unit type shape if no frames (shouldn't happen in practice)
                <() as facet_core::Facet>::SHAPE
            });

        let mut path = Path::new(root_shape);

        // Walk ALL frames, extracting navigation steps
        // This matches the storage path computation in end()
        let frames = self.frames();
        for (frame_idx, frame) in frames.iter().enumerate() {
            match &frame.tracker {
                Tracker::Struct {
                    current_child: Some(idx),
                    ..
                } => {
                    path.push(PathStep::Field(*idx as u32));
                }
                Tracker::Enum {
                    current_child: Some(idx),
                    ..
                } => {
                    path.push(PathStep::Field(*idx as u32));
                }
                Tracker::List {
                    current_child: Some(idx),
                    ..
                } => {
                    path.push(PathStep::Index(*idx as u32));
                }
                Tracker::Array {
                    current_child: Some(idx),
                    ..
                } => {
                    path.push(PathStep::Index(*idx as u32));
                }
                Tracker::Option {
                    building_inner: true,
                    ..
                } => {
                    // Option with building_inner contributes OptionSome to path
                    path.push(PathStep::OptionSome);
                }
                Tracker::SmartPointer {
                    building_inner: true,
                    ..
                } => {
                    // SmartPointer with building_inner contributes Deref to path
                    path.push(PathStep::Deref);
                }
                Tracker::SmartPointerSlice {
                    current_child: Some(idx),
                    ..
                } => {
                    // SmartPointerSlice with current_child contributes Index to path
                    path.push(PathStep::Index(*idx as u32));
                }
                Tracker::Inner {
                    building_inner: true,
                } => {
                    // Inner with building_inner contributes Inner to path
                    path.push(PathStep::Inner);
                }
                Tracker::Map {
                    current_entry_index: Some(idx),
                    building_key,
                    ..
                } => {
                    // Map with active entry contributes MapKey or MapValue with entry index
                    if *building_key {
                        path.push(PathStep::MapKey(*idx as u32));
                    } else {
                        path.push(PathStep::MapValue(*idx as u32));
                    }
                }
                // Other tracker types (Set, Result, etc.)
                // don't contribute to the storage path - they're transparent wrappers
                _ => {}
            }

            // If the next frame is a proxy frame, add a Proxy step (matches end())
            if frame_idx + 1 < frames.len() && frames[frame_idx + 1].using_custom_deserialization {
                path.push(PathStep::Proxy);
            }
        }

        path
    }
}

impl<'facet, const BORROW: bool> Drop for Partial<'facet, BORROW> {
    fn drop(&mut self) {
        trace!("🧹 Partial is being dropped");

        // With the ownership transfer model:
        // - When we enter a field, parent's iset[idx] is cleared
        // - Parent won't try to drop fields with iset[idx] = false
        // - No double-free possible by construction

        // 1. Clean up stored frames from deferred state
        if let FrameMode::Deferred {
            stored_frames,
            stack,
            ..
        } = &mut self.mode
        {
            // Stored frames have ownership of their data (parent's iset was cleared).
            // IMPORTANT: Process in deepest-first order so children are dropped before parents.
            // Child frames have data pointers into parent memory, so parents must stay valid
            // until all their children are cleaned up.
            //
            // CRITICAL: Before dropping a child frame, we must mark the parent's field as
            // uninitialized. Otherwise, when we later drop the parent, it will try to drop
            // that field again, causing a double-free.
            let mut stored_frames = core::mem::take(stored_frames);
            let mut paths: Vec<_> = stored_frames.keys().cloned().collect();
            // Sort by path depth (number of steps), deepest first
            paths.sort_by_key(|p| core::cmp::Reverse(p.steps.len()));
            for path in paths {
                if let Some(mut frame) = stored_frames.remove(&path) {
                    // Before dropping this frame, update the parent to prevent double-free.
                    // The parent path is everything except the last step.
                    let parent_path = Path {
                        shape: path.shape,
                        steps: path.steps[..path.steps.len().saturating_sub(1)].to_vec(),
                    };

                    // Helper to find parent frame in stored_frames or stack
                    let find_parent_frame =
                        |stored: &mut alloc::collections::BTreeMap<Path, Frame>,
                         stk: &mut [Frame],
                         pp: &Path|
                         -> Option<*mut Frame> {
                            if let Some(pf) = stored.get_mut(pp) {
                                Some(pf as *mut Frame)
                            } else {
                                let idx = pp.steps.len();
                                stk.get_mut(idx).map(|f| f as *mut Frame)
                            }
                        };

                    match path.steps.last() {
                        Some(PathStep::Field(field_idx)) => {
                            let field_idx = *field_idx as usize;
                            if let Some(parent_ptr) =
                                find_parent_frame(&mut stored_frames, stack, &parent_path)
                            {
                                // SAFETY: parent_ptr is valid for the duration of this block
                                let parent_frame = unsafe { &mut *parent_ptr };
                                match &mut parent_frame.tracker {
                                    Tracker::Struct { iset, .. } => {
                                        iset.unset(field_idx);
                                    }
                                    Tracker::Enum { data, .. } => {
                                        data.unset(field_idx);
                                    }
                                    _ => {}
                                }
                            }
                        }
                        // Stored map key/value frames keep their TrackedBuffer ownership
                        // until consume-time in finish_deferred; pending_entries is only
                        // populated by the walk, so at Partial::drop it contains nothing
                        // referencing these frames. Their own deinit/dealloc handles the
                        // partial-init drop. No parent-side cleanup required.
                        Some(PathStep::Index(_)) => {
                            // List element frames with RopeSlot ownership are handled by
                            // the deinit check for RopeSlot - they skip dropping since the
                            // rope owns the data. No parent update needed.
                        }
                        _ => {}
                    }
                    frame.deinit_for_cancellation();
                    frame.dealloc();
                }
            }
        }

        // 2. Pop and deinit stack frames
        // CRITICAL: Before deiniting a child frame, we must mark the parent's field as
        // uninitialized. Otherwise, the parent will try to drop the field again.
        loop {
            let stack = self.mode.stack_mut();
            if stack.is_empty() {
                break;
            }

            let mut frame = stack.pop().unwrap();

            // If this frame has Field ownership, mark the parent's bit as unset
            // so the parent won't try to drop it again.
            if let FrameOwnership::Field { field_idx } = frame.ownership
                && let Some(parent_frame) = stack.last_mut()
            {
                match &mut parent_frame.tracker {
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

            frame.deinit_for_cancellation();
            frame.dealloc();
        }
    }
}

#[cfg(test)]
mod size_tests {
    use super::*;
    use core::mem::size_of;

    #[test]
    fn print_type_sizes() {
        eprintln!("\n=== Type Sizes ===");
        eprintln!("Frame: {} bytes", size_of::<Frame>());
        eprintln!("Tracker: {} bytes", size_of::<Tracker>());
        eprintln!("ISet: {} bytes", size_of::<ISet>());
        eprintln!("AllocatedShape: {} bytes", size_of::<AllocatedShape>());
        eprintln!("FrameOwnership: {} bytes", size_of::<FrameOwnership>());
        eprintln!("PtrUninit: {} bytes", size_of::<facet_core::PtrUninit>());
        eprintln!("Option<usize>: {} bytes", size_of::<Option<usize>>());
        eprintln!(
            "Option<&'static facet_core::ProxyDef>: {} bytes",
            size_of::<Option<&'static facet_core::ProxyDef>>()
        );
        eprintln!(
            "TypePlanNode: {} bytes",
            size_of::<typeplan::TypePlanNode>()
        );
        eprintln!("Vec<Frame>: {} bytes", size_of::<Vec<Frame>>());
        eprintln!("MapInsertState: {} bytes", size_of::<MapInsertState>());
        eprintln!(
            "DynamicValueState: {} bytes",
            size_of::<DynamicValueState>()
        );
        eprintln!("===================\n");
    }
}

#[cfg(test)]
mod pending_smart_pointer_tests {
    use alloc::{borrow::Cow, string::String};
    use core::{
        mem::MaybeUninit,
        sync::atomic::{AtomicUsize, Ordering},
    };

    use super::*;
    use facet::Facet;

    static EXTERNAL_PENDING_DROPS: AtomicUsize = AtomicUsize::new(0);

    #[derive(Clone, Facet)]
    struct ExternalPendingDropTracker;

    impl Drop for ExternalPendingDropTracker {
        fn drop(&mut self) {
            EXTERNAL_PENDING_DROPS.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn cow_pending_smart_pointer_drop_uses_actual_staging_shape() {
        let mut partial = Partial::alloc::<Cow<'static, str>>().unwrap();
        let string_layout = String::SHAPE.layout.sized_layout().unwrap();
        let staging_data = facet_core::alloc_for_layout(string_layout);
        unsafe {
            staging_data.put(String::from(
                "the pending String staging allocation must be dropped as String, not str",
            ));
        }

        let root = partial.frames_mut().last_mut().unwrap();
        root.tracker = Tracker::SmartPointer {
            building_inner: false,
            pending_inner: Some(PendingSmartPointerInner {
                data: staging_data,
                allocated: AllocatedShape::new(String::SHAPE, string_layout.size()),
                ownership: FrameOwnership::Owned,
            }),
        };
        // This models the old deferred state marker and verifies that pending
        // staging prevents `Frame::deinit` from dropping uninitialized Cow storage.
        root.is_init = true;

        drop(partial);
    }

    #[test]
    fn external_reinitialization_drains_direct_pending_smart_pointer_staging() {
        EXTERNAL_PENDING_DROPS.store(0, Ordering::SeqCst);

        let mut destination = MaybeUninit::<Cow<'static, ExternalPendingDropTracker>>::uninit();
        let destination = PtrUninit::new(destination.as_mut_ptr().cast::<u8>());
        let mut partial: Partial<'_, false> = unsafe {
            Partial::from_raw_with_shape(destination, Cow::<ExternalPendingDropTracker>::SHAPE)
                .unwrap()
        };

        let staging_layout = ExternalPendingDropTracker::SHAPE
            .layout
            .sized_layout()
            .unwrap();
        let staging_data = facet_core::alloc_for_layout(staging_layout);
        unsafe { staging_data.put(ExternalPendingDropTracker) };

        let root = partial.frames_mut().last_mut().unwrap();
        root.tracker = Tracker::SmartPointer {
            building_inner: false,
            pending_inner: Some(PendingSmartPointerInner {
                data: staging_data,
                allocated: AllocatedShape::new(
                    ExternalPendingDropTracker::SHAPE,
                    staging_layout.size(),
                ),
                ownership: FrameOwnership::Owned,
            }),
        };

        partial.prepare_for_reinitialization();
        assert_eq!(EXTERNAL_PENDING_DROPS.load(Ordering::SeqCst), 1);

        drop(partial);
        assert_eq!(EXTERNAL_PENDING_DROPS.load(Ordering::SeqCst), 1);
    }
}
