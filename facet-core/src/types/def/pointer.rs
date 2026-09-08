use crate::{PtrConst, PtrMut, PtrUninit, Shape, bitflags};

/// Describes a pointer — including a vtable to query and alter its state,
/// and the inner shape (the pointee type in the pointer).
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct PointerDef {
    /// vtable for interacting with the pointer
    pub vtable: &'static PointerVTable,

    /// shape of the inner type of the pointer, if not opaque
    pub pointee: Option<&'static Shape>,

    /// shape of the corresponding strong pointer, if this pointer is weak
    ///
    /// the layer of indirection is to break the strong <-> weak reference cycle,
    /// since consts may not have cycles in their definitions.
    pub weak: Option<fn() -> &'static Shape>,

    /// shape of the corresponding weak pointer, if this pointer is strong
    pub strong: Option<&'static Shape>,

    /// Flags representing various characteristics of the pointer
    pub flags: PointerFlags,

    /// An optional field to identify the kind of pointer
    pub known: Option<KnownPointer>,
}

impl PointerDef {
    /// Returns shape of the inner type of the pointer, if not opaque
    pub const fn pointee(&self) -> Option<&'static Shape> {
        self.pointee
    }

    /// Returns shape of the corresponding strong pointer, if this pointer is weak
    pub fn weak(&self) -> Option<&'static Shape> {
        self.weak.map(|f| f())
    }

    /// Returns shape of the corresponding weak pointer, if this pointer is strong
    pub const fn strong(&self) -> Option<&'static Shape> {
        self.strong
    }

    /// Whether a new pointer can be constructed as an independent owner of its pointee.
    ///
    /// Most pointer types transfer the pointee with [`NewIntoFn`]. Pointer types
    /// such as [`Cow`](alloc::borrow::Cow) can instead borrow the pointee and
    /// immediately materialize an owned representation through the paired
    /// `borrow_from_pointee_fn` and `promote_to_owned_fn` operations.
    pub const fn constructible_from_pointee(&self) -> bool {
        self.vtable.new_into_fn.is_some()
            || (self.vtable.borrow_from_pointee_fn.is_some()
                && self.vtable.promote_to_owned_fn.is_some())
            || matches!(
                self.known,
                Some(KnownPointer::Box | KnownPointer::Rc | KnownPointer::Arc)
            )
    }
}

bitflags! {
    /// Flags to represent various characteristics of pointers
    pub struct PointerFlags: u8 {
        /// An empty set of flags
        const EMPTY = 0;

        /// Whether the pointer is weak (like `std::sync::Weak`)
        const WEAK = 1 << 0;
        /// Whether the pointer is atomic (like `std::sync::Arc`)
        const ATOMIC = 1 << 1;
        /// Whether the pointer is a lock (like `std::sync::Mutex`)
        const LOCK = 1 << 2;
    }
}

/// Tries to upgrade the weak pointer to a strong one.
///
/// If the upgrade succeeds, initializes the pointer into the given `strong`, and returns a
/// copy of `strong`, which has been guaranteed to be initialized. If the upgrade fails, `None` is
/// returned and `strong` is not initialized.
///
/// `weak` is not moved out of.
///
/// # Safety
///
/// `weak` must be a valid weak pointer (like [`alloc::sync::Weak`] or [`alloc::rc::Weak`]).
///
/// `strong` must be allocated, and of the right layout for the corresponding pointer.
///
/// `strong` must not have been initialized yet.
pub type UpgradeIntoFn = unsafe fn(weak: PtrMut, strong: PtrUninit) -> Option<PtrMut>;

/// Downgrades a strong pointer to a weak one.
///
/// Initializes the pointer into the given `weak`, and returns a copy of `weak`, which has
/// been guaranteed to be initialized.
///
/// Only strong pointers can be downgraded (like [`alloc::sync::Arc`] or [`alloc::rc::Rc`]).
///
/// # Safety
///
/// `strong` must be a valid strong pointer (like [`alloc::sync::Arc`] or [`alloc::rc::Rc`]).
///
/// `weak` must be allocated, and of the right layout for the corresponding weak pointer.
///
/// `weak` must not have been initialized yet.
pub type DowngradeIntoFn = unsafe fn(strong: PtrMut, weak: PtrUninit) -> PtrMut;

/// Tries to obtain a reference to the inner value of the pointer.
///
/// This can only be used with strong pointers (like [`alloc::sync::Arc`] or [`alloc::rc::Rc`]).
///
/// # Safety
///
/// `this` must be a valid strong pointer (like [`alloc::sync::Arc`] or [`alloc::rc::Rc`]).
pub type BorrowFn = unsafe extern "C" fn(this: PtrConst) -> PtrConst;

/// Creates a new pointer wrapping the given value.
///
/// Initializes the pointer into the given `this`, and returns a copy of `this`, which has
/// been guaranteed to be initialized.
///
/// This can only be used with strong pointers (like [`alloc::sync::Arc`] or [`alloc::rc::Rc`]).
///
/// # Safety
///
/// `this` must be allocated, and of the right layout for the corresponding pointer.
///
/// `this` must not have been initialized yet.
///
/// `ptr` must point to a value of type `T`.
///
/// `ptr` is moved out of (with [`core::ptr::read`]) — it should be deallocated afterwards (e.g.
/// with [`core::mem::forget`]) but NOT dropped).
pub type NewIntoFn = unsafe extern "C" fn(this: PtrUninit, ptr: PtrMut) -> PtrMut;

/// Constructs a borrowed pointer from a stable initialized pointee.
///
/// This initializes `dst` as a pointer value. The pointee is not consumed.
///
/// # Safety
///
/// - `dst` must be allocated, have the right layout for the pointer type, and
///   be uninitialized.
/// - `pointee` must point to a valid value of the pointer's reflected pointee
///   type.
/// - The caller must keep `pointee` valid and stable for every use through the
///   constructed pointer and any derived references, copies, or clones. Any
///   lifetime represented by those values must be bounded by the source's
///   validity, including when the caller manages that bound dynamically.
/// - Dropping or promoting the constructed pointer only ends that instance's
///   borrow; it does not end borrows retained by other values.
/// - The implementation must initialize `dst`; the initialized value may
///   access `pointee` until it is dropped or promoted.
pub type BorrowFromPointeeFn = unsafe extern "C" fn(dst: PtrUninit, pointee: PtrConst);

/// Promotes a borrowed pointer to its owned representation in place.
///
/// On return, `this` remains initialized and does not access a pointee that it
/// borrowed before the call. For a pointer already in its owned representation,
/// promotion may be a no-op.
///
/// This operation affects only `this`. References or borrowed pointer clones
/// obtained before promotion may still depend on the original pointee.
///
/// # Safety
///
/// - `this` must point to an initialized value of the reflected pointer type.
/// - The caller must have exclusive access to `this`.
/// - If `this` currently borrows a pointee, that pointee must remain valid and
///   stable until this function returns.
/// - The caller may drop or deallocate a formerly borrowed pointee only when
///   no other value can still access it. In particular, any borrowed copies,
///   clones, or derived references must also have ended their dependency.
pub type PromoteToOwnedFn = unsafe extern "C" fn(this: PtrMut);

/// Type-erased result of locking a mutex-like or reader-writer lock pointer.
///
/// The type parameter `P` determines the capability of the returned pointer:
/// - `PtrConst` for read locks (shared access)
/// - `PtrMut` for write/mutex locks (exclusive access)
pub struct LockResult<P> {
    /// The data that was locked
    data: P,
    /// The guard that protects the data
    guard: PtrConst,
    /// The vtable for the guard
    guard_vtable: &'static LockGuardVTable,
}

/// Result of acquiring a read lock (shared access) - data is immutable
pub type ReadLockResult = LockResult<PtrConst>;

/// Result of acquiring a write/mutex lock (exclusive access) - data is mutable
pub type WriteLockResult = LockResult<PtrMut>;

impl<P> LockResult<P> {
    /// Creates a new `LockResult` from its components.
    ///
    /// # Safety
    ///
    /// - `data` must point to valid data protected by the guard
    /// - `guard` must point to a valid guard that, when dropped via `guard_vtable.drop_in_place`,
    ///   will release the lock
    /// - The guard must outlive any use of `data`
    #[must_use]
    pub const unsafe fn new(
        data: P,
        guard: PtrConst,
        guard_vtable: &'static LockGuardVTable,
    ) -> Self {
        Self {
            data,
            guard,
            guard_vtable,
        }
    }

    /// Returns a reference to the locked data
    #[must_use]
    pub const fn data(&self) -> &P {
        &self.data
    }
}

impl WriteLockResult {
    /// Returns a const pointer to the locked data (convenience for write locks)
    #[must_use]
    pub const fn data_const(&self) -> PtrConst {
        self.data.as_const()
    }
}

impl<P> Drop for LockResult<P> {
    fn drop(&mut self) {
        unsafe {
            (self.guard_vtable.drop_in_place)(self.guard);
        }
    }
}

/// Functions for manipulating a guard
pub struct LockGuardVTable {
    /// Drops the guard in place
    pub drop_in_place: unsafe fn(guard: PtrConst),
}

/// Acquires a lock on a mutex-like pointer (exclusive access)
pub type LockFn = unsafe fn(opaque: PtrConst) -> Result<WriteLockResult, ()>;

/// Acquires a read lock on a reader-writer lock-like pointer (shared access)
pub type ReadFn = unsafe fn(opaque: PtrConst) -> Result<ReadLockResult, ()>;

/// Acquires a write lock on a reader-writer lock-like pointer (exclusive access)
pub type WriteFn = unsafe fn(opaque: PtrConst) -> Result<WriteLockResult, ()>;

/// Creates a new slice builder for a pointer: ie. for `Arc<[u8]>`, it builds a
/// `Vec<u8>` internally, to which you can push, and then turn into an `Arc<[u8]>` at
/// the last stage.
///
/// This works for any `U` in `Arc<[U]>` because those have separate concrete implementations, and
/// thus, separate concrete vtables.
pub type SliceBuilderNewFn = fn() -> PtrMut;

/// Pushes a value into a slice builder.
///
/// # Safety
///
/// Item must point to a valid value of type `U` and must be initialized.
/// Function is infallible as it uses the global allocator.
/// `item` is moved out of (with [`core::ptr::read`]) — it should be deallocated afterwards
/// but NOT dropped.
/// Note: `item` must be PtrMut (not PtrConst) because ownership is transferred.
pub type SliceBuilderPushFn = unsafe fn(builder: PtrMut, item: PtrMut);

/// Converts a slice builder into a pointer. This takes ownership of the builder
/// and frees the backing storage.
///
/// # Safety
///
/// The builder must be valid and must not be used after this function is called.
/// Function is infallible as it uses the global allocator.
pub type SliceBuilderConvertFn = unsafe fn(builder: PtrMut) -> PtrConst;

/// Frees a slice builder without converting it into a pointer
///
/// # Safety
///
/// The builder must be valid and must not be used after this function is called.
pub type SliceBuilderFreeFn = unsafe fn(builder: PtrMut);

/// Functions for creating and manipulating slice builders.
#[derive(Debug, Clone, Copy)]
pub struct SliceBuilderVTable {
    /// See [`SliceBuilderNewFn`]
    pub new_fn: SliceBuilderNewFn,
    /// See [`SliceBuilderPushFn`]
    pub push_fn: SliceBuilderPushFn,
    /// See [`SliceBuilderConvertFn`]
    pub convert_fn: SliceBuilderConvertFn,
    /// See [`SliceBuilderFreeFn`]
    pub free_fn: SliceBuilderFreeFn,
}

impl SliceBuilderVTable {
    /// Const ctor for slice builder vtable; all hooks required.
    #[must_use]
    pub const fn new(
        new_fn: SliceBuilderNewFn,
        push_fn: SliceBuilderPushFn,
        convert_fn: SliceBuilderConvertFn,
        free_fn: SliceBuilderFreeFn,
    ) -> Self {
        Self {
            new_fn,
            push_fn,
            convert_fn,
            free_fn,
        }
    }
}

/// Functions for interacting with a pointer
#[derive(Debug, Clone, Copy)]
pub struct PointerVTable {
    /// See [`UpgradeIntoFn`]
    pub upgrade_into_fn: Option<UpgradeIntoFn>,

    /// See [`DowngradeIntoFn`]
    pub downgrade_into_fn: Option<DowngradeIntoFn>,

    /// See [`BorrowFn`]
    pub borrow_fn: Option<BorrowFn>,

    /// See [`NewIntoFn`]
    pub new_into_fn: Option<NewIntoFn>,

    /// See [`BorrowFromPointeeFn`]
    pub borrow_from_pointee_fn: Option<BorrowFromPointeeFn>,

    /// See [`PromoteToOwnedFn`]
    pub promote_to_owned_fn: Option<PromoteToOwnedFn>,

    /// See [`LockFn`]
    pub lock_fn: Option<LockFn>,

    /// See [`ReadFn`]
    pub read_fn: Option<ReadFn>,

    /// See [`WriteFn`]
    pub write_fn: Option<WriteFn>,

    /// See [`SliceBuilderVTable`]
    pub slice_builder_vtable: Option<&'static SliceBuilderVTable>,
}

impl Default for PointerVTable {
    fn default() -> Self {
        Self::new()
    }
}

impl PointerVTable {
    /// Const ctor with all entries set to `None`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            upgrade_into_fn: None,
            downgrade_into_fn: None,
            borrow_fn: None,
            new_into_fn: None,
            borrow_from_pointee_fn: None,
            promote_to_owned_fn: None,
            lock_fn: None,
            read_fn: None,
            write_fn: None,
            slice_builder_vtable: None,
        }
    }
}

/// Represents common standard library pointer kinds
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum KnownPointer {
    /// [`Box<T>`](alloc::boxed::Box), heap-allocated values with single ownership
    Box,
    /// [`Rc<T>`](alloc::rc::Rc), reference-counted values with multiple ownership
    Rc,
    /// [`Weak<T>`](alloc::rc::Weak), a weak reference to an `Rc`-managed value
    RcWeak,
    /// [`Arc<T>`](alloc::sync::Arc), thread-safe reference-counted values with multiple ownership
    Arc,
    /// [`Weak<T>`](alloc::sync::Weak), a weak reference to an `Arc`-managed value
    ArcWeak,
    /// [`Cow<'a, T>`](alloc::borrow::Cow), a clone-on-write smart pointer
    Cow,
    /// [`Pin<P>`](core::pin::Pin), a type that pins values behind a pointer
    Pin,
    /// [`Cell<T>`](core::cell::Cell), a mutable memory location with interior mutability
    Cell,
    /// [`RefCell<T>`](core::cell::RefCell), a mutable memory location with dynamic borrowing rules
    RefCell,
    /// [`OnceCell<T>`](core::cell::OnceCell), a cell that can be written to only once
    OnceCell,
    /// `Mutex<T>`, a mutual exclusion primitive (requires std)
    Mutex,
    /// `RwLock<T>`, a reader-writer lock (requires std)
    RwLock,
    /// `OnceLock<T>`, a cell that can be written to only once (requires std)
    OnceLock,
    /// `LazyLock<T, F>`, a lazy-initialized value (requires std)
    LazyLock,
    /// [`NonNull<T>`](core::ptr::NonNull), a wrapper around a raw pointer that is not null
    NonNull,
    /// `&T`
    SharedReference,
    /// `&mut T`
    ExclusiveReference,
}
