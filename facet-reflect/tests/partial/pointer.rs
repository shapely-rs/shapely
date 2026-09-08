use std::{
    borrow::Cow,
    mem::MaybeUninit,
    ptr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
};

use facet::{Facet, KnownPointer, PtrMut, PtrUninit, Type, UserType};
use facet_reflect::{Partial, Peek};
use facet_testhelpers::{IPanic, test};

#[derive(Debug, PartialEq, Facet)]
struct Inner {
    value: i32,
}

#[derive(Debug, PartialEq, Facet)]
struct OuterYesArc {
    inner: Arc<Inner>,
}

#[derive(Debug, PartialEq, Facet)]
struct OuterNoArc {
    inner: Inner,
}

static COW_DROP_TRACKER_DROPS: AtomicUsize = AtomicUsize::new(0);
static COW_DROP_TRACKER_TEST_LOCK: Mutex<()> = Mutex::new(());
static COW_ENUM_DROPS: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone, Debug, Facet, PartialEq)]
struct CowDropTracker {
    value: u8,
}

#[derive(Clone, Debug, Facet, PartialEq)]
#[repr(u8)]
enum CowDropTrackerEnum {
    Unit,
    Payload {
        value: CowDropTracker,
    },
    Pair {
        first: CowDropTracker,
        second: CowDropTracker,
    },
}

impl Drop for CowDropTrackerEnum {
    fn drop(&mut self) {
        COW_ENUM_DROPS.fetch_add(1, Ordering::SeqCst);
    }
}

// A deliberately small model of Cloud Terrastodon's Object Explorer. The
// selected value is type-erased, while the request field is found from the
// request shape rather than referenced directly by the construction code.
#[derive(Clone, Debug, PartialEq, Facet)]
#[repr(C)]
struct OrganizationUrl {
    value: String,
}

#[derive(Debug, Facet)]
#[repr(C)]
struct ListProjectsRequest {
    organization_url: Cow<'static, OrganizationUrl>,
}

impl Drop for CowDropTracker {
    fn drop(&mut self) {
        COW_DROP_TRACKER_DROPS.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn cow_object_explorer_can_borrow_then_materialize_a_selected_field() {
    let selected_value = OrganizationUrl {
        value: "https://dev.azure.com/example".to_owned(),
    };
    let selected = Peek::new(&selected_value);

    let Type::User(UserType::Struct(request_def)) = ListProjectsRequest::SHAPE.ty else {
        panic!("ListProjectsRequest should have reflected struct metadata");
    };
    let organization_url_field = request_def
        .fields
        .iter()
        .find(|field| field.name == "organization_url")
        .expect("Object Explorer should find the selected request field");
    let organization_url_shape = organization_url_field.shape();
    let organization_url_pointer = organization_url_shape
        .def
        .into_pointer()
        .expect("organization_url should be reflected as a pointer");

    assert_eq!(
        organization_url_pointer
            .pointee()
            .expect("Cow should expose its pointee shape")
            .id,
        selected.shape().id,
        "the dynamically selected value must match the Cow field's pointee type"
    );
    assert_eq!(organization_url_pointer.known, Some(KnownPointer::Cow));

    let borrow_from_pointee = organization_url_pointer
        .vtable
        .borrow_from_pointee_fn
        .expect("Cow should expose reflected borrowed construction");
    let promote_to_owned = organization_url_pointer
        .vtable
        .promote_to_owned_fn
        .expect("Cow should expose reflected in-place ownership promotion");

    // The request has one field in this miniature model, so initializing the
    // reflected field initializes the complete request.
    let mut request = MaybeUninit::<ListProjectsRequest>::uninit();
    let organization_url_slot = unsafe {
        PtrUninit::from_maybe_uninit(&mut request).field_uninit(organization_url_field.offset)
    };
    unsafe { borrow_from_pointee(organization_url_slot, selected.data()) };
    let mut request = unsafe { request.assume_init() };

    assert!(matches!(
        &request.organization_url,
        Cow::Borrowed(value) if ptr::eq(*value, &selected_value),
    ));

    // Object Explorer may now release the selected source: promotion must
    // remove the Cow's dependency on it before that happens. `Peek` is a Copy
    // view, so only the source itself owns storage here.
    unsafe { promote_to_owned(PtrMut::new(&mut request.organization_url)) };
    drop(selected_value);

    assert!(matches!(
        &request.organization_url,
        Cow::Owned(value) if value.value == "https://dev.azure.com/example",
    ));
}

#[test]
fn cow_partial_sized_borrow_promote_drops_source_once() -> Result<(), IPanic> {
    let _lock = COW_DROP_TRACKER_TEST_LOCK.lock().unwrap();
    COW_DROP_TRACKER_DROPS.store(0, Ordering::SeqCst);

    let cow = Partial::alloc::<Cow<'static, CowDropTracker>>()?
        .begin_smart_ptr()?
        .set(CowDropTracker { value: 41 })?
        .end()?
        .build()?
        .materialize::<Cow<'static, CowDropTracker>>()?;

    assert!(matches!(&cow, Cow::Owned(value) if value.value == 41));
    assert_eq!(
        COW_DROP_TRACKER_DROPS.load(Ordering::SeqCst),
        1,
        "the temporary pointee is dropped after Cow materializes its owned clone"
    );

    drop(cow);
    assert_eq!(COW_DROP_TRACKER_DROPS.load(Ordering::SeqCst), 2);
    Ok(())
}

fn check_cow_enum_source_and_owned_clone_drop(deferred: bool) -> Result<(), IPanic> {
    let _lock = COW_DROP_TRACKER_TEST_LOCK.lock().unwrap();

    for variant in ["Unit", "Payload"] {
        COW_ENUM_DROPS.store(0, Ordering::SeqCst);
        COW_DROP_TRACKER_DROPS.store(0, Ordering::SeqCst);
        let mut partial = Partial::alloc::<Cow<'static, CowDropTrackerEnum>>()?;
        if deferred {
            partial = partial.begin_deferred()?;
        }
        partial = partial.begin_smart_ptr()?.select_variant_named(variant)?;
        if variant == "Payload" {
            partial = partial
                .begin_field("value")?
                .set(CowDropTracker { value: 46 })?
                .end()?;
        }
        partial = partial.end()?;
        if deferred {
            partial = partial.finish_deferred()?;
        }
        let cow = partial
            .build()?
            .materialize::<Cow<'static, CowDropTrackerEnum>>()?;

        assert!(matches!(&cow, Cow::Owned(_)));
        assert_eq!(COW_ENUM_DROPS.load(Ordering::SeqCst), 1);
        let payload_drops = usize::from(variant == "Payload");
        assert_eq!(COW_DROP_TRACKER_DROPS.load(Ordering::SeqCst), payload_drops);

        drop(cow);
        assert_eq!(COW_ENUM_DROPS.load(Ordering::SeqCst), 2);
        assert_eq!(
            COW_DROP_TRACKER_DROPS.load(Ordering::SeqCst),
            payload_drops * 2
        );
    }
    Ok(())
}

#[test]
fn cow_partial_enum_borrow_promote_drops_source_and_owned_clone() -> Result<(), IPanic> {
    check_cow_enum_source_and_owned_clone_drop(false)
}

#[test]
fn cow_partial_enum_borrow_promote_deferred_drops_source_and_owned_clone() -> Result<(), IPanic> {
    check_cow_enum_source_and_owned_clone_drop(true)
}

#[test]
fn cow_partial_enum_cancellation_respects_initialized_fields() -> Result<(), IPanic> {
    let _lock = COW_DROP_TRACKER_TEST_LOCK.lock().unwrap();

    for deferred in [false, true] {
        for store_inner in [false, true] {
            if store_inner && !deferred {
                continue; // Strict end() promotes instead of storing staging.
            }
            for variant in ["Unit", "Payload", "Pair"] {
                COW_ENUM_DROPS.store(0, Ordering::SeqCst);
                COW_DROP_TRACKER_DROPS.store(0, Ordering::SeqCst);
                let mut partial = Partial::alloc::<Cow<'static, CowDropTrackerEnum>>()?;
                if deferred {
                    partial = partial.begin_deferred()?;
                }
                partial = partial.begin_smart_ptr()?.select_variant_named(variant)?;
                if variant != "Unit" {
                    let field = if variant == "Payload" {
                        "value"
                    } else {
                        "first"
                    };
                    partial = partial
                        .begin_field(field)?
                        .set(CowDropTracker { value: 47 })?
                        .end()?;
                }
                if store_inner {
                    partial = partial.end()?;
                }
                drop(partial);

                // Deferred field frames are canceled separately, leaving the enum
                // incomplete. Only a unit variant or fully assembled strict payload
                // can run its whole-value destructor. Pair never has both fields.
                let enum_drops =
                    usize::from(variant == "Unit" || (!deferred && variant == "Payload"));
                assert_eq!(COW_ENUM_DROPS.load(Ordering::SeqCst), enum_drops);
                assert_eq!(
                    COW_DROP_TRACKER_DROPS.load(Ordering::SeqCst),
                    usize::from(variant != "Unit")
                );
            }
        }
    }
    Ok(())
}

#[test]
fn cow_str_begin_smart_ptr_materializes_owned() -> Result<(), IPanic> {
    let cow = Partial::alloc::<Cow<'static, str>>()?
        .begin_smart_ptr()?
        .set(String::from("borrow then own"))?
        .end()?
        .build()?
        .materialize::<Cow<'static, str>>()?;

    assert!(matches!(&cow, Cow::Owned(value) if value == "borrow then own"));
    Ok(())
}

#[test]
fn cow_partial_sized_borrow_promote_deferred() -> Result<(), IPanic> {
    let _lock = COW_DROP_TRACKER_TEST_LOCK.lock().unwrap();
    COW_DROP_TRACKER_DROPS.store(0, Ordering::SeqCst);

    let partial = Partial::alloc::<Cow<'static, CowDropTracker>>()?
        .begin_deferred()?
        .begin_smart_ptr()?
        .set(CowDropTracker { value: 42 })?
        .end()?
        .finish_deferred()?;
    let cow = partial
        .build()?
        .materialize::<Cow<'static, CowDropTracker>>()?;

    assert!(matches!(&cow, Cow::Owned(value) if value.value == 42));
    assert_eq!(COW_DROP_TRACKER_DROPS.load(Ordering::SeqCst), 1);
    drop(cow);
    assert_eq!(COW_DROP_TRACKER_DROPS.load(Ordering::SeqCst), 2);
    Ok(())
}

#[test]
fn cow_partial_sized_deferred_drop_without_finish_drops_staging_source() -> Result<(), IPanic> {
    let _lock = COW_DROP_TRACKER_TEST_LOCK.lock().unwrap();
    COW_DROP_TRACKER_DROPS.store(0, Ordering::SeqCst);

    let partial = Partial::alloc::<Cow<'static, CowDropTracker>>()?
        .begin_deferred()?
        .begin_smart_ptr()?
        .set(CowDropTracker { value: 43 })?
        .end()?;
    drop(partial);

    assert_eq!(COW_DROP_TRACKER_DROPS.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn cow_external_deferred_reentry_drops_replaced_staging_source() -> Result<(), IPanic> {
    let _lock = COW_DROP_TRACKER_TEST_LOCK.lock().unwrap();
    COW_DROP_TRACKER_DROPS.store(0, Ordering::SeqCst);

    let mut destination = MaybeUninit::<Cow<'static, CowDropTracker>>::uninit();
    let destination = PtrUninit::new(destination.as_mut_ptr().cast::<u8>());

    // Cloud Terrastodon uses caller-owned destinations for some dynamic object
    // construction. Re-entering the same deferred pointer must clean the first
    // staging value before the second one replaces it.
    let partial: Partial<'_, false> =
        unsafe { Partial::from_raw_with_shape(destination, Cow::<CowDropTracker>::SHAPE)? };
    let partial = partial
        .begin_deferred()?
        .begin_smart_ptr()?
        .set(CowDropTracker { value: 44 })?
        .end()?
        .begin_smart_ptr()?
        .set(CowDropTracker { value: 45 })?
        .end()?;
    drop(partial);

    assert_eq!(COW_DROP_TRACKER_DROPS.load(Ordering::SeqCst), 2);
    Ok(())
}

#[test]
fn cow_str_deferred_drop_without_finish_releases_staging_string() -> Result<(), IPanic> {
    let partial = Partial::alloc::<Cow<'static, str>>()?
        .begin_deferred()?
        .begin_smart_ptr()?
        .set(String::from(
            "the deferred String staging allocation must be released on cancellation",
        ))?
        .end()?;
    drop(partial);

    // Miri verifies that the non-empty String allocation was released.
    Ok(())
}

#[test]
fn cow_str_begin_smart_ptr_deferred_materializes_owned() -> Result<(), IPanic> {
    let partial = Partial::alloc::<Cow<'static, str>>()?
        .begin_deferred()?
        .begin_smart_ptr()?
        .set(String::from("deferred borrow then own"))?
        .end()?
        .finish_deferred()?;
    let cow = partial.build()?.materialize::<Cow<'static, str>>()?;

    assert!(matches!(&cow, Cow::Owned(value) if value == "deferred borrow then own"));
    Ok(())
}

#[test]
fn cow_str_deferred_reentry_restores_string_staging() -> Result<(), IPanic> {
    let partial = Partial::alloc::<Cow<'static, str>>()?
        .begin_deferred()?
        .begin_smart_ptr()?
        .set(String::from("first staged value"))?
        .end()?
        .begin_smart_ptr()?
        .set(String::from("replacement staged value"))?
        .end()?
        .finish_deferred()?;
    let cow = partial.build()?.materialize::<Cow<'static, str>>()?;

    assert!(matches!(&cow, Cow::Owned(value) if value == "replacement staged value"));
    Ok(())
}

#[test]
fn outer_no_arc() {
    let mut partial: Partial<'_> = Partial::alloc::<OuterNoArc>().unwrap();
    partial = partial.begin_field("inner").unwrap();
    partial = partial.begin_field("value").unwrap();
    partial = partial.set(1234_i32).unwrap();
    partial = partial.end().unwrap();
    partial = partial.end().unwrap();
    let o = partial
        .build()
        .unwrap()
        .materialize::<OuterNoArc>()
        .unwrap();
    assert_eq!(
        o,
        OuterNoArc {
            inner: Inner { value: 1234 }
        }
    );
}

#[test]
fn outer_yes_arc_put() {
    let mut partial: Partial<'_> = Partial::alloc::<OuterYesArc>().unwrap();
    let inner = Arc::new(Inner { value: 5678 });
    partial = partial.begin_field("inner").unwrap();
    partial = partial.set(inner.clone()).unwrap();
    partial = partial.end().unwrap();
    let o = partial
        .build()
        .unwrap()
        .materialize::<OuterYesArc>()
        .unwrap();
    assert_eq!(o, OuterYesArc { inner });
}

#[test]
fn outer_yes_arc_pointee() {
    let mut partial: Partial<'_> = Partial::alloc::<OuterYesArc>().unwrap();
    partial = partial.begin_field("inner").unwrap();
    partial = partial.begin_smart_ptr().unwrap();
    partial = partial.begin_field("value").unwrap();
    partial = partial.set(4321_i32).unwrap();
    partial = partial.end().unwrap();
    partial = partial.end().unwrap();
    partial = partial.end().unwrap();
    let o = partial
        .build()
        .unwrap()
        .materialize::<OuterYesArc>()
        .unwrap();
    assert_eq!(
        o,
        OuterYesArc {
            inner: Arc::new(Inner { value: 4321 })
        }
    );
}

#[test]
fn outer_yes_arc_field_named_twice_error() {
    let mut partial: Partial<'_> = Partial::alloc::<OuterYesArc>().unwrap();
    partial = partial.begin_field("inner").unwrap();
    // Try to do begin_field again instead of begin_smart_ptr; this should error
    let err = partial.begin_field("value").err().unwrap();
    let err_string = format!("{err}");
    assert!(
        err_string.contains("opaque types cannot be reflected upon"),
        "Error message should mention 'opaque types cannot be reflected upon', got: {err_string}"
    );
}

#[test]
fn arc_str_begin_smart_ptr_good() {
    let mut partial = Partial::alloc::<Arc<str>>().unwrap();
    partial = partial.begin_smart_ptr().unwrap();
    partial = partial.set(String::from("foobar")).unwrap();
    partial = partial.end().unwrap();
    let built = partial.build().unwrap().materialize::<Arc<str>>().unwrap();
    assert_eq!(&*built, "foobar");
}

#[test]
fn arc_str_begin_smart_ptr_bad_1() {
    let partial = Partial::alloc::<Arc<str>>().unwrap();
    let _err = partial.build().unwrap_err();
    #[cfg(not(miri))]
    insta::assert_snapshot!(_err);
}

#[test]
fn arc_str_begin_smart_ptr_bad_2a() {
    let mut partial = Partial::alloc::<Arc<str>>().unwrap();
    partial = partial.begin_smart_ptr().unwrap();
    let _err = partial.build().unwrap_err();
    #[cfg(not(miri))]
    insta::assert_snapshot!(_err);
}

#[test]
fn arc_str_begin_smart_ptr_bad_2b() {
    let mut partial = Partial::alloc::<Arc<str>>().unwrap();
    partial = partial.begin_smart_ptr().unwrap();
    let _err = match partial.end() {
        Ok(_) => panic!("expected error"),
        Err(e) => e,
    };
    #[cfg(not(miri))]
    insta::assert_snapshot!(_err);
}

#[test]
fn arc_str_begin_smart_ptr_bad_3() {
    let mut partial = Partial::alloc::<Arc<str>>().unwrap();
    partial = partial.begin_smart_ptr().unwrap();
    partial = partial.set(String::from("foobar")).unwrap();
    let _err = partial.build().unwrap_err();
    #[cfg(not(miri))]
    insta::assert_snapshot!(_err);
}

#[test]
fn rc_str_begin_smart_ptr_once() {
    use std::rc::Rc;
    let mut partial = Partial::alloc::<Rc<str>>().unwrap();
    partial = partial.begin_smart_ptr().unwrap();
    partial = partial.set(String::from("foobar")).unwrap();
    partial = partial.end().unwrap();
    let built = partial
        .build()
        .unwrap()
        .materialize::<std::rc::Rc<str>>()
        .unwrap();
    assert_eq!(&*built, "foobar");
}

#[test]
fn rc_str_begin_smart_ptr_twice() -> Result<(), IPanic> {
    use std::rc::Rc;
    let mut partial = Partial::alloc::<Rc<str>>()?;

    eprintln!("==== first go");
    partial = partial.begin_smart_ptr()?;
    partial = partial.set(String::from("foobar"))?;
    partial = partial.end()?;

    eprintln!("==== second go");
    partial = partial.begin_smart_ptr()?;
    partial = partial.set(String::from("barbaz"))?;
    partial = partial.end()?;

    eprintln!("==== build");
    let built = partial.build()?.materialize::<std::rc::Rc<str>>()?;
    assert_eq!(&*built, "barbaz");

    Ok(())
}

#[test]
fn box_str_begin_smart_ptr() {
    let mut partial = Partial::alloc::<Box<str>>().unwrap();
    partial = partial.begin_smart_ptr().unwrap();
    partial = partial.set(String::from("foobar")).unwrap();
    partial = partial.end().unwrap();
    let built = partial.build().unwrap().materialize::<Box<str>>().unwrap();
    assert_eq!(&*built, "foobar");
}

#[test]
fn arc_slice_u8_begin_smart_ptr_good() {
    {
        // Just to make sure: Vec<u8> construction works
        let mut partial = Partial::alloc::<Vec<u8>>().unwrap();
        partial = partial.init_list().unwrap();
        partial = partial.push(2_u8).unwrap();
        partial = partial.push(3_u8).unwrap();
        partial = partial.push(4_u8).unwrap();
        let built = partial.build().unwrap().materialize::<Vec<u8>>().unwrap();
        assert_eq!(&*built, &[2, 3, 4]);
    }

    {
        // Now, does Arc<[u8]> work.unwrap()
        let mut partial = Partial::alloc::<Arc<[u8]>>().unwrap();
        partial = partial.begin_smart_ptr().unwrap();
        partial = partial.init_list().unwrap();
        partial = partial.push(2_u8).unwrap();
        partial = partial.push(3_u8).unwrap();
        partial = partial.push(4_u8).unwrap();
        partial = partial.end().unwrap();
        let built = partial.build().unwrap().materialize::<Arc<[u8]>>().unwrap();
        assert_eq!(&*built, &[2, 3, 4]);
    }
}

// =============================================================================
// Tests migrated from src/partial/tests.rs
// =============================================================================

#[cfg(not(miri))]
macro_rules! assert_snapshot {
    ($($tt:tt)*) => {
        insta::assert_snapshot!($($tt)*)
    };
}
#[cfg(miri)]
macro_rules! assert_snapshot {
    ($($tt:tt)*) => {{ let _ = $($tt)*; }};
}

#[test]
fn box_init() -> Result<(), IPanic> {
    let hv = Partial::alloc::<Box<u32>>()?
        // Push into the Box to build its inner value
        .begin_smart_ptr()?
        .set(42u32)?
        .end()?
        .build()?
        .materialize::<Box<u32>>()?;
    assert_eq!(*hv, 42);
    Ok(())
}

#[test]
fn box_partial_init() -> Result<(), IPanic> {
    // Don't initialize the Box at all
    assert_snapshot!(Partial::alloc::<Box<u32>>()?.build().unwrap_err());
    Ok(())
}

#[test]
fn box_struct() -> Result<(), IPanic> {
    #[derive(Facet, Debug, PartialEq)]
    struct Point {
        x: f64,
        y: f64,
    }

    let hv = Partial::alloc::<Box<Point>>()?
        // Push into the Box
        .begin_smart_ptr()?
        // Build the Point inside the Box using set_field shorthand
        .set_field("x", 1.0)?
        .set_field("y", 2.0)?
        // end from Box
        .end()?
        .build()?
        .materialize::<Box<Point>>()?;
    assert_eq!(*hv, Point { x: 1.0, y: 2.0 });
    Ok(())
}

#[test]
fn drop_box_partially_initialized() -> Result<(), IPanic> {
    use core::sync::atomic::{AtomicUsize, Ordering};
    static BOX_DROP_COUNT: AtomicUsize = AtomicUsize::new(0);
    static INNER_DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

    #[derive(Facet, Debug)]
    struct DropCounter {
        value: u32,
    }

    impl Drop for DropCounter {
        fn drop(&mut self) {
            INNER_DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            println!("Dropping DropCounter with value: {}", self.value);
        }
    }

    BOX_DROP_COUNT.store(0, Ordering::SeqCst);
    INNER_DROP_COUNT.store(0, Ordering::SeqCst);

    {
        let mut partial = Partial::alloc::<Box<DropCounter>>()?;

        // Initialize the Box's inner value using set
        partial = partial.begin_smart_ptr()?;
        partial = partial.set(DropCounter { value: 99 })?;
        let _partial = partial.end()?;

        // Drop the partial - should drop the Box which drops the inner value
    }

    assert_eq!(
        INNER_DROP_COUNT.load(Ordering::SeqCst),
        1,
        "Should drop the inner value through Box's drop"
    );
    Ok(())
}

#[test]
fn arc_init() -> Result<(), IPanic> {
    let hv = Partial::alloc::<Arc<u32>>()?
        // Push into the Arc to build its inner value
        .begin_smart_ptr()?
        .set(42u32)?
        .end()?
        .build()?
        .materialize::<Arc<u32>>()?;
    assert_eq!(*hv, 42);
    Ok(())
}

#[test]
fn arc_partial_init() -> Result<(), IPanic> {
    // Don't initialize the Arc at all
    assert_snapshot!(Partial::alloc::<Arc<u32>>()?.build().unwrap_err());
    Ok(())
}

#[test]
fn arc_struct() -> Result<(), IPanic> {
    #[derive(Facet, Debug, PartialEq)]
    struct Point {
        x: f64,
        y: f64,
    }

    let hv = Partial::alloc::<Arc<Point>>()?
        // Push into the Arc
        .begin_smart_ptr()?
        // Build the Point inside the Arc using set_field shorthand
        .set_field("x", 3.0)?
        .set_field("y", 4.0)?
        // end from Arc
        .end()?
        .build()?
        .materialize::<Arc<Point>>()?;
    assert_eq!(*hv, Point { x: 3.0, y: 4.0 });
    Ok(())
}

#[test]
fn drop_arc_partially_initialized() -> Result<(), IPanic> {
    use core::sync::atomic::{AtomicUsize, Ordering};
    static INNER_DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

    #[derive(Facet, Debug)]
    struct DropCounter {
        value: u32,
    }

    impl Drop for DropCounter {
        fn drop(&mut self) {
            INNER_DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            println!("Dropping DropCounter with value: {}", self.value);
        }
    }

    INNER_DROP_COUNT.store(0, Ordering::SeqCst);

    {
        let mut partial = Partial::alloc::<Arc<DropCounter>>()?;

        // Initialize the Arc's inner value
        partial = partial.begin_smart_ptr()?;
        partial = partial.set(DropCounter { value: 123 })?;
        let _partial = partial.end()?;

        // Drop the partial - should drop the Arc which drops the inner value
    }

    assert_eq!(
        INNER_DROP_COUNT.load(Ordering::SeqCst),
        1,
        "Should drop the inner value through Arc's drop"
    );
    Ok(())
}
