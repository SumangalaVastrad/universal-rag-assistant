import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import _classify_document_type, _validate_upload_permission

def test_upload_governance():
    print("--- Testing Departmental Upload Permission Governance ---")

    # 1. Staff / General user tests
    allowed_staff_pub, _ = _validate_upload_permission("Staff / General User", "public")
    assert allowed_staff_pub == True, "Staff should be allowed to upload public docs"

    allowed_staff_hr, msg = _validate_upload_permission("Staff / General User", "hr")
    assert allowed_staff_hr == False, "Staff should NOT be allowed to upload HR docs"
    print("[OK] Staff uploading HR doc blocked")

    allowed_staff_fin, msg = _validate_upload_permission("Staff / General User", "finance")
    assert allowed_staff_fin == False, "Staff should NOT be allowed to upload Finance docs"
    print("[OK] Staff uploading Finance doc blocked")

    # 2. HR Department Head tests
    allowed_hr_hr, _ = _validate_upload_permission("HR Department Head", "hr")
    assert allowed_hr_hr == True, "HR head should be allowed to upload HR docs"

    allowed_hr_fin, msg = _validate_upload_permission("HR Department Head", "finance")
    assert allowed_hr_fin == False, "HR head should NOT be allowed to upload Finance docs"
    print("[OK] HR head uploading Finance doc blocked")

    # 3. Finance Department Head tests
    allowed_fin_fin, _ = _validate_upload_permission("Finance Department Head", "finance")
    assert allowed_fin_fin == True, "Finance head should be allowed to upload Finance docs"

    allowed_fin_hr, msg = _validate_upload_permission("Finance Department Head", "hr")
    assert allowed_fin_hr == False, "Finance head should NOT be allowed to upload HR docs"
    print("[OK] Finance head uploading HR doc blocked")

    # 4. Admin tests
    for doc in ["admin", "hr", "finance", "public"]:
        allowed_admin, _ = _validate_upload_permission("Admin (Full Access)", doc)
        assert allowed_admin == True, f"Admin should be allowed to upload {doc} docs"
    print("[OK] Admin can upload all document types (Admin, HR, Finance, Public)")

    print("--- ALL UPLOAD GOVERNANCE TESTS PASSED! ---")

if __name__ == "__main__":
    test_upload_governance()
