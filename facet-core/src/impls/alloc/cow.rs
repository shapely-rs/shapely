use crate::{
    Def, Facet, KnownPointer, OxPtrConst, OxPtrMut, OxPtrUninit, PointerDef, PointerFlags,
    PointerVTable, PtrConst, PtrMut, PtrUninit, Shape, ShapeBuilder, Type, TypeNameFn,
    TypeNameOpts, TypeOpsIndirect, TypeParam, UserType, VTableIndirect, Variance, VarianceDep,
    VarianceDesc,
};
use alloc::borrow::Cow;
use alloc::borrow::ToOwned;

/// Debug for `Cow<T>` - delegates to inner T's debug
///
/// # Safety
/// The pointer must point to a valid Cow<'_, T> value
unsafe fn cow_debug<T: ?Sized + ToOwned + 'static>(
    ox: OxPtrConst,
    f: &mut core::fmt::Formatter<'_>,
) -> Option<core::fmt::Result>
where
    T::Owned: 'static,
{
    let cow_ref: &Cow<'_, T> = unsafe { ox.get::<Cow<'static, T>>() };

    // Get T's shape from the Cow's shape
    let cow_shape = ox.shape();
    let t_shape = cow_shape.inner?;

    let inner_ref: &T = cow_ref.as_ref();

    let inner_ptr = PtrConst::new(inner_ref as *const T);
    unsafe { t_shape.call_debug(inner_ptr, f) }
}

/// Display for `Cow<T>` - delegates to inner T's display if available
///
/// # Safety
/// The pointer must point to a valid Cow<'_, T> value
unsafe fn cow_display<T: ?Sized + ToOwned + 'static>(
    ox: OxPtrConst,
    f: &mut core::fmt::Formatter<'_>,
) -> Option<core::fmt::Result>
where
    T::Owned: 'static,
{
    let cow_ref: &Cow<'_, T> = unsafe { ox.get::<Cow<'static, T>>() };

    // Get T's shape from the Cow's shape
    let cow_shape = ox.shape();
    let t_shape = cow_shape.inner?;

    if !t_shape.vtable.has_display() {
        return None;
    }

    let inner_ref: &T = cow_ref.as_ref();
    let inner_ptr = PtrConst::new(inner_ref as *const T);

    unsafe { t_shape.call_display(inner_ptr, f) }
}

/// PartialEq for `Cow<T>`
///
/// # Safety
/// Both pointers must point to valid Cow<'_, T> values
unsafe fn cow_partial_eq<T: ?Sized + ToOwned + 'static>(
    a: OxPtrConst,
    b: OxPtrConst,
) -> Option<bool>
where
    T::Owned: 'static,
{
    let a_cow_ref: &Cow<'_, T> = unsafe { a.get::<Cow<'static, T>>() };
    let b_cow_ref: &Cow<'_, T> = unsafe { b.get::<Cow<'static, T>>() };

    let cow_shape = a.shape();
    let t_shape = cow_shape.inner?;

    let a_inner = PtrConst::new(a_cow_ref.as_ref() as *const T);
    let b_inner = PtrConst::new(b_cow_ref.as_ref() as *const T);

    unsafe { t_shape.call_partial_eq(a_inner, b_inner) }
}

/// PartialOrd for `Cow<T>`
///
/// # Safety
/// Both pointers must point to valid Cow<'_, T> values
unsafe fn cow_partial_cmp<T: ?Sized + ToOwned + 'static>(
    a: OxPtrConst,
    b: OxPtrConst,
) -> Option<Option<core::cmp::Ordering>>
where
    T::Owned: 'static,
{
    let a_cow_ref: &Cow<'_, T> = unsafe { a.get::<Cow<'static, T>>() };
    let b_cow_ref: &Cow<'_, T> = unsafe { b.get::<Cow<'static, T>>() };

    let cow_shape = a.shape();
    let t_shape = cow_shape.inner?;

    let a_inner = PtrConst::new(a_cow_ref.as_ref() as *const T);
    let b_inner = PtrConst::new(b_cow_ref.as_ref() as *const T);

    unsafe { t_shape.call_partial_cmp(a_inner, b_inner) }
}

/// Ord for `Cow<T>`
///
/// # Safety
/// Both pointers must point to valid Cow<'_, T> values
unsafe fn cow_cmp<T: ?Sized + ToOwned + 'static>(
    a: OxPtrConst,
    b: OxPtrConst,
) -> Option<core::cmp::Ordering>
where
    T::Owned: 'static,
{
    let a_cow_ref: &Cow<'_, T> = unsafe { a.get::<Cow<'static, T>>() };
    let b_cow_ref: &Cow<'_, T> = unsafe { b.get::<Cow<'static, T>>() };

    let cow_shape = a.shape();
    let t_shape = cow_shape.inner?;

    let a_inner = PtrConst::new(a_cow_ref.as_ref() as *const T);
    let b_inner = PtrConst::new(b_cow_ref.as_ref() as *const T);

    unsafe { t_shape.call_cmp(a_inner, b_inner) }
}

/// Borrow the inner value from `Cow<T>`
///
/// # Safety
/// `this` must point to a valid Cow<'_, T> value
unsafe extern "C" fn cow_borrow<T: ?Sized + ToOwned + 'static>(this: PtrConst) -> PtrConst
where
    T::Owned: 'static,
{
    // SAFETY: Same layout reasoning as cow_debug
    let cow_ref: &Cow<'_, T> =
        unsafe { &*(this.as_byte_ptr() as *const alloc::borrow::Cow<'_, T>) };
    let inner_ref: &T = cow_ref.as_ref();
    PtrConst::new(inner_ref as *const T)
}

/// Construct `Cow::Borrowed` from a stable pointee.
///
/// # Safety
///
/// The caller must keep `pointee` valid for the resulting Cow and every derived
/// borrow or clone. Promoting one Cow only ends that instance's dependency.
unsafe extern "C" fn cow_borrow_from_pointee<T: ?Sized + ToOwned + 'static>(
    dst: PtrUninit,
    pointee: PtrConst,
) where
    T::Owned: 'static,
{
    let pointee = unsafe { pointee.get::<T>() };
    unsafe { dst.put(Cow::<'static, T>::Borrowed(pointee)) };
}

/// Materialize an owned Cow in place.
unsafe extern "C" fn cow_promote_to_owned<T: ?Sized + ToOwned + 'static>(this: PtrMut)
where
    T::Owned: 'static,
{
    let cow = unsafe { this.as_mut::<Cow<'static, T>>() };
    let _ = cow.to_mut();
}

unsafe impl<'a, T> Facet<'a> for Cow<'a, T>
where
    T: 'a + ?Sized + ToOwned + 'static,
    T: Facet<'a>,
    T::Owned: Facet<'static>,
{
    const SHAPE: &'static Shape = &const {
        const fn build_cow_vtable<T: ?Sized + ToOwned + 'static>() -> VTableIndirect
        where
            T::Owned: Facet<'static> + 'static,
        {
            VTableIndirect {
                debug: Some(cow_debug::<T>),
                display: Some(cow_display::<T>),
                partial_eq: Some(cow_partial_eq::<T>),
                partial_cmp: Some(cow_partial_cmp::<T>),
                cmp: Some(cow_cmp::<T>),
                ..VTableIndirect::EMPTY
            }
        }

        const fn build_cow_type_ops<'facet, T>() -> TypeOpsIndirect
        where
            T: ?Sized + ToOwned + 'static + Facet<'facet>,
            T::Owned: Facet<'static> + 'static,
        {
            unsafe fn drop_in_place<T: ?Sized + ToOwned + 'static>(ox: OxPtrMut)
            where
                T::Owned: 'static,
            {
                unsafe {
                    core::ptr::drop_in_place(
                        ox.ptr().as_ptr::<Cow<'static, T>>() as *mut Cow<'static, T>
                    )
                };
            }

            unsafe fn clone_into<T: ?Sized + ToOwned + 'static>(src: OxPtrConst, dst: OxPtrMut)
            where
                T::Owned: 'static,
            {
                let src_cow_ref: &Cow<'_, T> = unsafe { src.get::<Cow<'static, T>>() };
                let cloned = src_cow_ref.clone();
                // IMPORTANT: `clone_into` must be valid for writes to potentially-uninitialized
                // destination memory. Do not create `&mut Cow` here (that would assume initialization
                // and the assignment would drop garbage).
                let out: *mut Cow<'static, T> =
                    unsafe { dst.ptr().as_ptr::<Cow<'static, T>>() as *mut Cow<'static, T> };
                unsafe { core::ptr::write(out, cloned) };
            }

            /// Default for `Cow<T>` - creates `Cow::Owned(T::Owned::default())`
            /// by checking if T::Owned supports default at runtime via its shape.
            ///
            /// # Safety
            /// dst must be valid for writes
            unsafe fn default_in_place<T: ?Sized + ToOwned + 'static>(dst: OxPtrUninit) -> bool
            where
                T::Owned: Facet<'static> + 'static,
            {
                // Get the Owned type's shape from the second type param
                let cow_shape = dst.shape();
                let type_params = cow_shape.type_params;
                if type_params.len() < 2 {
                    return false;
                }

                let owned_shape = type_params[1].shape;

                // Allocate space for T::Owned and call default_in_place
                let owned_layout = match owned_shape.layout.sized_layout() {
                    Ok(layout) => layout,
                    Err(_) => return false,
                };

                let owned_uninit = crate::alloc_for_layout(owned_layout);
                if unsafe { owned_shape.call_default_in_place(owned_uninit) }.is_none() {
                    // Default not supported, deallocate and return
                    unsafe { crate::dealloc_for_layout(owned_uninit.assume_init(), owned_layout) };
                    return false;
                }

                // Move the constructed T::Owned out of the temporary allocation.
                // This leaves the allocation uninitialized, so we must deallocate the backing storage.
                let owned_value: T::Owned =
                    unsafe { core::ptr::read(owned_uninit.as_byte_ptr() as *const T::Owned) };
                unsafe { crate::dealloc_for_layout(owned_uninit.assume_init(), owned_layout) };

                // Write the Cow::Owned to uninitialized memory
                unsafe { dst.put(Cow::<'static, T>::Owned(owned_value)) };
                true
            }

            unsafe fn truthy<'facet, T>(ptr: PtrConst) -> bool
            where
                T: ?Sized + ToOwned + 'static + Facet<'facet>,
                T::Owned: Facet<'static> + 'static,
            {
                let cow_ref: &Cow<'_, T> = unsafe { ptr.get::<Cow<'static, T>>() };
                let inner_shape = <T as Facet<'facet>>::SHAPE;
                if let Some(truthy) = inner_shape.truthiness_fn() {
                    let inner: &T = cow_ref.as_ref();
                    unsafe { truthy(PtrConst::new(inner as *const T)) }
                } else {
                    false
                }
            }

            TypeOpsIndirect {
                drop_in_place: drop_in_place::<T>,
                default_in_place: Some(default_in_place::<T>),
                clone_into: Some(clone_into::<T>),
                is_truthy: Some(truthy::<'facet, T>),
            }
        }

        const fn build_type_name<'a, T: Facet<'a> + ?Sized + ToOwned>() -> TypeNameFn {
            fn type_name_impl<'a, T: Facet<'a> + ?Sized + ToOwned>(
                _shape: &Shape,
                f: &mut core::fmt::Formatter<'_>,
                opts: TypeNameOpts,
            ) -> core::fmt::Result {
                write!(f, "Cow")?;
                if let Some(opts) = opts.for_children() {
                    write!(f, "<")?;
                    T::SHAPE.write_type_name(f, opts)?;
                    write!(f, ">")?;
                } else {
                    write!(f, "<…>")?;
                }
                Ok(())
            }
            type_name_impl::<T>
        }

        ShapeBuilder::for_sized::<Cow<'a, T>>("Cow")
            .module_path("alloc::borrow")
            .type_name(build_type_name::<T>())
            .ty(Type::User(UserType::Opaque))
            .def(Def::Pointer(PointerDef {
                vtable: &const {
                    PointerVTable {
                        borrow_fn: Some(cow_borrow::<T>),
                        borrow_from_pointee_fn: Some(cow_borrow_from_pointee::<T>),
                        promote_to_owned_fn: Some(cow_promote_to_owned::<T>),
                        ..PointerVTable::new()
                    }
                },
                pointee: Some(T::SHAPE),
                weak: None,
                strong: None,
                flags: PointerFlags::EMPTY,
                known: Some(KnownPointer::Cow),
            }))
            .type_params(&[
                TypeParam {
                    name: "T",
                    shape: T::SHAPE,
                },
                TypeParam {
                    name: "Owned",
                    shape: <T::Owned>::SHAPE,
                },
            ])
            .inner(T::SHAPE)
            // `enum Cow<'a, B> { Borrowed(&'a B), Owned(<B as ToOwned>::Owned) }`.
            //
            // The base MUST be `Covariant`: the `Borrowed` arm holds a real `&'a B`,
            // so `Cow` carries a lifetime of its own. Declaring `base: Bivariant` with
            // only a covariant dep on `T` is the easy mistake here — it makes
            // `Cow<'a, str>` report `Bivariant` (because `str` is bivariant), which is
            // precisely the use-after-free this commit fixes.
            //
            // Both parameters are in *invariant* position: rustc says "the enum
            // `Cow<'a, B>` is invariant over the parameter `B`", because
            // `<B as ToOwned>::Owned` is an associated-type projection.
            // `VarianceDep::invariant` passes `Bivariant` through unchanged, so the
            // overwhelmingly common `Cow<'a, str>` still lands on `Covariant`.
            .variance(VarianceDesc {
                base: Variance::Covariant,
                deps: &const {
                    [
                        VarianceDep::invariant(T::SHAPE),
                        VarianceDep::invariant(<T::Owned>::SHAPE),
                    ]
                },
            })
            .vtable_indirect(&const { build_cow_vtable::<T>() })
            .type_ops_indirect(&const { build_cow_type_ops::<'a, T>() })
            .build()
    };
}

#[cfg(test)]
mod tests {
    use alloc::string::String;

    use super::*;

    #[test]
    fn test_cow_type_params() {
        let [type_param_1, type_param_2] = <Cow<'_, str>>::SHAPE.type_params else {
            panic!("Cow<'_, T> should only have 2 type params")
        };
        assert_eq!(type_param_1.shape(), str::SHAPE);
        assert_eq!(type_param_2.shape(), String::SHAPE);
    }

    #[test]
    fn test_cow_vtable_borrow_and_promote_str() {
        facet_testhelpers::setup();

        let cow_shape = <Cow<'static, str>>::SHAPE;
        let cow_def = cow_shape
            .def
            .into_pointer()
            .expect("Cow<'static, str> should have a smart pointer definition");

        assert!(
            cow_def.vtable.new_into_fn.is_none(),
            "Cow's borrowed construction must not masquerade as an owned-pointee move"
        );

        let borrow_from_pointee = cow_def
            .vtable
            .borrow_from_pointee_fn
            .expect("Cow<'static, str> should expose borrowed construction");
        let promote_to_owned = cow_def
            .vtable
            .promote_to_owned_fn
            .expect("Cow<'static, str> should expose in-place ownership promotion");
        let borrow_fn = cow_def
            .vtable
            .borrow_fn
            .expect("Cow<'static, str> should expose inner borrowing");

        let source = String::from("example");
        let cow_uninit = cow_shape.allocate().unwrap();
        unsafe {
            borrow_from_pointee(cow_uninit, PtrConst::new(source.as_str() as *const str));
        }
        let cow_ptr = unsafe { cow_uninit.assume_init() };

        let borrowed = unsafe { cow_ptr.get::<Cow<'static, str>>() };
        assert!(matches!(borrowed, Cow::Borrowed(value) if *value == "example"));

        unsafe { promote_to_owned(cow_ptr) };
        let promoted = unsafe { cow_ptr.get::<Cow<'static, str>>() };
        assert!(matches!(promoted, Cow::Owned(value) if value == "example"));

        drop(source);
        let borrowed_ptr = unsafe { borrow_fn(cow_ptr.as_const()) };
        assert_eq!(unsafe { borrowed_ptr.get::<str>() }, "example");

        unsafe {
            cow_shape
                .call_drop_in_place(cow_ptr)
                .expect("promoted Cow should be droppable");
            cow_shape.deallocate_mut(cow_ptr).unwrap();
        }
    }

    #[test]
    fn test_cow_vtable_promotion_preserves_other_clones_borrows() {
        let cow_def = <Cow<'static, str>>::SHAPE.def.into_pointer().unwrap();
        let borrow_from_pointee = cow_def.vtable.borrow_from_pointee_fn.unwrap();
        let promote_to_owned = cow_def.vtable.promote_to_owned_fn.unwrap();
        let source = String::from("shared borrowed source");
        let mut destination = core::mem::MaybeUninit::<Cow<'static, str>>::uninit();

        // The source remains live until both independently borrowed Cows have
        // been promoted, and no derived references escape this scope.
        unsafe {
            borrow_from_pointee(
                PtrUninit::from_maybe_uninit(&mut destination),
                PtrConst::new(source.as_str() as *const str),
            );
        }
        let mut original = unsafe { destination.assume_init() };
        let mut cloned = original.clone();

        unsafe { promote_to_owned(PtrMut::new(&mut original)) };
        assert!(matches!(&original, Cow::Owned(_)));
        assert!(matches!(
            &cloned,
            Cow::Borrowed(value) if core::ptr::eq(*value, source.as_str()),
        ));
        assert_eq!(cloned, "shared borrowed source");

        unsafe { promote_to_owned(PtrMut::new(&mut cloned)) };
        drop(source);
        assert!(matches!(&cloned, Cow::Owned(_)));
        assert_eq!(original, cloned);
    }
}
