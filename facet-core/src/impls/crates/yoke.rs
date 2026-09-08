#![cfg(feature = "yoke")]

use core::{mem::MaybeUninit, ops::Deref};
use stable_deref_trait::StableDeref;
use yoke::{Yoke, Yokeable};

use crate::{
    Facet, OxPtrConst, OxPtrMut, OxPtrUninit, PtrConst, PtrMut, PtrUninit, ShapeBuilder,
    TryFromOutcome, Type, TypeNameOpts, TypeOpsIndirect, UserType, VTableDirect, VTableErased,
    VTableIndirect, Variance, VarianceDesc,
};

// Helper functions to create type_name formatters
fn type_name_yoke<'a, Y, C>(
    _shape: &crate::Shape,
    f: &mut core::fmt::Formatter<'_>,
    opts: TypeNameOpts,
) -> core::fmt::Result
where
    Y: for<'y> Yokeable<'y>,
    C: Facet<'a> + StableDeref,
    <C as Deref>::Target: Facet<'a>,
    <Y as Yokeable<'a>>::Output: Facet<'a>,
{
    write!(f, "Yoke")?;
    if let Some(opts) = opts.for_children() {
        write!(f, "<")?;
        <Y as Yokeable<'a>>::Output::SHAPE.write_type_name(f, opts)?;
        write!(f, ", ")?;
        <C as Deref>::Target::SHAPE.write_type_name(f, opts)?;
        write!(f, ">")?;
    } else {
        write!(f, "<…>")?;
    }
    Ok(())
}

// VTableIndirect functions for Yoke<Y, C>
// Used for serialization via innermost_peek() - returns the cart's data for round-trip
unsafe fn yoke_try_borrow_inner<'a, Y, C>(
    ox: OxPtrConst,
) -> Option<Result<PtrMut, alloc::string::String>>
where
    Y: for<'y> Yokeable<'y>,
    C: Facet<'a> + StableDeref,
    <C as Deref>::Target: Facet<'a>,
    <Y as Yokeable<'a>>::Output: Facet<'a>,
{
    unsafe {
        let yoke: &Yoke<Y, C> = ox.ptr().get();
        // Return the cart's data, not the yoked value
        // This ensures serialization outputs the original format (e.g., "hello|yoke")
        // which can be deserialized back via builder_shape -> try_from
        let cart_ref: &<C as Deref>::Target = yoke.backing_cart().deref();
        Some(Ok(PtrMut::new(
            cart_ref as *const <C as Deref>::Target as *mut <C as Deref>::Target,
        )))
    }
}

// Type operations for Yoke<Y, C>
unsafe fn yoke_drop<Y: for<'y> Yokeable<'y>, C>(ox: OxPtrMut) {
    unsafe { core::ptr::drop_in_place(ox.ptr().as_ptr::<Yoke<Y, C>>() as *mut Yoke<Y, C>) };
}

/// try_from: Convert from cart (C) to Yoke<Y, C>
///
/// This function is called during deserialization when the cart has been fully
/// constructed and we need to wrap it in a Yoke. The Yoke attaches to the cart
/// and constructs Y by borrowing from it.
///
/// # Strategy
/// 1. First tries Y's `borrow_from_pointee_fn` when its pointee matches the cart
///    target (for pointer types like Cow)
/// 2. Then tries Y's `try_from` (for user types with #[facet(from_ref)])
/// 3. Returns Unsupported if neither is available
#[allow(non_snake_case)]
unsafe fn yoke_try_from<'f, Y, C>(
    dst: OxPtrUninit,
    src_shape: &'static crate::Shape,
    src_ptr: PtrConst,
) -> TryFromOutcome
where
    Y: for<'y> Yokeable<'y>,
    C: Facet<'f> + StableDeref,
    <C as Deref>::Target: Facet<'f> + 'static,
    <Y as Yokeable<'f>>::Output: Facet<'f>,
{
    // Only accept the cart type
    if src_shape.id != C::SHAPE.id {
        return TryFromOutcome::Unsupported;
    }

    let CART_REF_SHAPE = <&<C as Deref>::Target as Facet>::SHAPE;
    let OUTPUT_SHAPE = <<Y as Yokeable>::Output as Facet>::SHAPE;

    unsafe {
        // Read the cart from source (consumes ownership)
        // Use try_attach_to_cart so we can return errors properly
        // First try: Y can construct a borrow from the stable cart pointee.
        let result = {
            if let Ok(ptr_def) = OUTPUT_SHAPE.def.into_pointer()
                && let Some(pointee_shape) = ptr_def.pointee
                && pointee_shape.is_shape(<C as Deref>::Target::SHAPE)
                && let Some(borrow_from_pointee_fn) = ptr_def.vtable.borrow_from_pointee_fn
            {
                Yoke::<Y, C>::try_attach_to_cart(src_ptr.read::<C>(), |cart_ref| {
                    let mut maybe_uninit = MaybeUninit::<Y::Output>::uninit();
                    borrow_from_pointee_fn(
                        PtrUninit::from_maybe_uninit(&mut maybe_uninit),
                        PtrConst::new(cart_ref as *const _),
                    );
                    // Read as Y::Output (same layout as Y, different lifetime)
                    let out = maybe_uninit.assume_init();
                    Ok(out)
                })
            } else {
                // Second try: Y has try_from in VTableDirect (user type with #[facet(from_ref)])
                match OUTPUT_SHAPE.vtable {
                    VTableErased::Direct(VTableDirect {
                        try_from: Some(try_from_fn),
                        ..
                    }) => {
                        Yoke::<Y, C>::try_attach_to_cart(src_ptr.read::<C>(), |cart_ref| {
                            let mut maybe_uninit = MaybeUninit::<Y::Output>::uninit();
                            let dst_ptr = maybe_uninit.as_mut_ptr() as *mut ();
                            // Use PtrConst::new with the unsized type - it handles wide pointers correctly
                            let cart_ptr = PtrConst::new(cart_ref as *const _);
                            match try_from_fn(dst_ptr, CART_REF_SHAPE, cart_ptr) {
                                TryFromOutcome::Converted => {
                                    // Read as Y::Output (same layout as Y, different lifetime)
                                    let out = maybe_uninit.assume_init();
                                    Ok(out)
                                }
                                e @ TryFromOutcome::Unsupported => {
                                    // Here we retain ownership of the source (maybe_uninit),
                                    // but we we shouldn't need to do anything since MaybeUninit doesn't need to be dropped.
                                    Err(e)
                                }
                                e @ TryFromOutcome::Failed(_) => Err(e),
                            }
                        })
                    }
                    VTableErased::Indirect(VTableIndirect {
                        try_from: Some(try_from_fn), // unsafe fn(OxPtrUninit, &'static Shape, PtrConst) -> TryFromOutcome
                        ..
                    }) => {
                        Yoke::<Y, C>::try_attach_to_cart(src_ptr.read::<C>(), |cart_ref| {
                            let mut maybe_uninit = MaybeUninit::<Y::Output>::uninit();
                            let out_ptr = OxPtrUninit::new(
                                PtrUninit::from_maybe_uninit(&mut maybe_uninit),
                                <Y as Yokeable>::Output::SHAPE,
                            );
                            // Use PtrConst::new with the unsized type - it handles wide pointers correctly
                            let cart_ptr = PtrConst::new(cart_ref as *const _);

                            let outcome = try_from_fn(out_ptr, CART_REF_SHAPE, cart_ptr);
                            match outcome {
                                TryFromOutcome::Converted => {
                                    // Read as Y::Output (same layout as Y, different lifetime)
                                    let out = maybe_uninit.assume_init();
                                    Ok(out)
                                }
                                e @ TryFromOutcome::Unsupported => {
                                    // Here we retain ownership of the source (maybe_uninit),
                                    // but we we shouldn't need to do anything since MaybeUninit doesn't need to be dropped.
                                    Err(e)
                                }
                                e @ TryFromOutcome::Failed(_) => Err(e),
                            }
                        })
                    }
                    // The borrow hook was unavailable, and no supported try_from path remained.
                    _ => Err(TryFromOutcome::Unsupported),
                }
            }
        };

        match result {
            Ok(yoke) => {
                dst.put(yoke);
                TryFromOutcome::Converted
            }
            Err(e) => e,
        }
    }
}

unsafe impl<'f, Y, C> Facet<'f> for Yoke<Y, C>
where
    Y: for<'y> Yokeable<'y> + Facet<'f>,
    C: Facet<'f> + StableDeref,
    <C as Deref>::Target: Facet<'f> + 'static,
    <Y as Yokeable<'f>>::Output: Facet<'f>,
{
    const SHAPE: &'static crate::Shape = &const {
        ShapeBuilder::for_sized::<Self>("Yoke")
            .module_path("yoke")
            .type_name(type_name_yoke::<Y, C>)
            .vtable_indirect(
                &const {
                    VTableIndirect {
                        // Used for serialization via innermost_peek()
                        try_borrow_inner: Some(yoke_try_borrow_inner::<Y, C>),
                        // Used for deserialization: converts cart (C) to Yoke<Y, C>
                        try_from: Some(yoke_try_from::<Y, C>),
                        ..VTableIndirect::EMPTY
                    }
                },
            )
            .type_ops_indirect(
                &const {
                    TypeOpsIndirect {
                        drop_in_place: yoke_drop::<Y, C>,
                        default_in_place: None,
                        clone_into: None,
                        is_truthy: None,
                    }
                },
            )
            .ty(Type::User(UserType::Opaque))
            // Not defined as Def::Pointer - we want innermost_peek to use try_borrow_inner
            // which returns the cart's data for proper round-trip serialization
            // builder_shape is the cart type - deserializers will build this first,
            // then use try_from to convert it to Yoke<Y, C>
            .builder_shape(C::SHAPE)
            .type_params(&[
                crate::TypeParam {
                    name: "Y",
                    shape: Y::SHAPE,
                },
                crate::TypeParam {
                    name: "C",
                    shape: C::SHAPE,
                },
            ])
            // inner is the cart's pointee - serialization via try_borrow_inner returns this
            .inner(<C as Deref>::Target::SHAPE)
            // Yoke<Y, C>'s variance is invariant
            .variance(VarianceDesc {
                base: Variance::Invariant,
                deps: &[],
            })
            .build()
    };
}

#[cfg(test)]
mod tests {
    use alloc::borrow::Cow;
    use alloc::sync::Arc;
    use core::mem::ManuallyDrop;

    use super::*;

    #[test]
    fn test_yoke_type_params() {
        let [type_param_1, type_param_2] = <Yoke<Cow<'static, str>, Arc<str>>>::SHAPE.type_params
        else {
            panic!("Yoke<T, U> should only have 2 type params")
        };
        assert_eq!(type_param_1.shape(), <Cow<'static, str>>::SHAPE);
        assert_eq!(type_param_2.shape(), <Arc::<str>>::SHAPE);
    }

    #[test]
    fn test_yoke_vtable_borrow_from_pointee_try_from_inner_drop() {
        facet_testhelpers::setup();

        let yoke_shape = <Yoke<Cow<'static, str>, Arc<str>>>::SHAPE;

        // Allocate memory for the Yoke
        let yoke_uninit_ptr = yoke_shape.allocate().unwrap();

        // Assert that it has a try_from
        assert!(yoke_shape.vtable.has_try_from());

        // Create the value and initialize the Yoke
        let value = ManuallyDrop::new(Arc::<str>::from("oui"));
        let res = unsafe {
            yoke_shape.call_try_from(
                <Arc<str>>::SHAPE,
                PtrConst::new_sized(&value as *const _),
                yoke_uninit_ptr,
            )
        }
        .expect("Should return Some since it has a try_from");
        assert_eq!(res, TryFromOutcome::Converted);
        let yoke_ptr = unsafe { yoke_uninit_ptr.assume_init() };

        // Borrow the inner value via try_borrow_inner
        // This returns the cart's data (&str), not the yoked value, for proper round-trip serialization
        let borrowed_inner_ptr = unsafe { yoke_shape.call_try_borrow_inner(yoke_ptr.as_const()) }
            .expect("try_borrow_inner should return Some")
            .expect("try_borrow_inner should succeed");

        // SAFETY: borrowed_ptr points to the cart's string data
        assert_eq!(unsafe { borrowed_inner_ptr.as_const().get::<str>() }, "oui");

        // Yoke is not defined as a pointer type (it's opaque) to ensure innermost_peek
        // uses try_borrow_inner for proper round-trip serialization
        assert!(
            yoke_shape.def.into_pointer().is_err(),
            "Yoke should not be a pointer type"
        );

        // Drop the Yoke in place
        // SAFETY: yoke_ptr points to a valid Yoke<Cow<'static, str>, Arc<str>>
        unsafe {
            yoke_shape
                .call_drop_in_place(yoke_ptr)
                .expect("Yoke<Cow<'static, str>, Arc<str>> should have drop_in_place");
        }

        // Deallocate the memory
        // SAFETY: arc_ptr was allocated by arc_shape and is now dropped (but memory is still valid)
        unsafe { yoke_shape.deallocate_mut(yoke_ptr).unwrap() };
    }

    #[test]
    fn test_yoke_rejects_borrowing_pointer_when_cart_pointee_does_not_match() {
        facet_testhelpers::setup();

        // `Cow<str>` has a borrowed-construction hook, but this cart contains a
        // `u8`. Calling that hook would reinterpret the cart target as `str`.
        let yoke_shape = <Yoke<Cow<'static, str>, Arc<u8>>>::SHAPE;
        let yoke_uninit_ptr = yoke_shape.allocate().unwrap();
        let mut value = ManuallyDrop::new(Arc::new(42_u8));

        let result = unsafe {
            yoke_shape.call_try_from(
                <Arc<u8>>::SHAPE,
                PtrConst::new_sized(&value as *const _),
                yoke_uninit_ptr,
            )
        }
        .expect("Yoke should expose try_from");

        assert_eq!(result, TryFromOutcome::Unsupported);

        // The unsupported path retains source ownership and leaves the target
        // uninitialized, so clean both up explicitly.
        unsafe {
            ManuallyDrop::drop(&mut value);
            yoke_shape.deallocate_uninit(yoke_uninit_ptr).unwrap();
        }
    }
}
