import pandas as pd
import streamlit as st

from core.camera_stream import release_inactive_cameras
from db.database import (
    create_employee,
    delete_employee,
    get_all_employees,
    get_employee_dataframe,
    initialize_database,
    update_employee,
    validate_employee_fields,
)

# st.set_page_config(page_title="Employees", layout="wide")

initialize_database()
release_inactive_cameras(st.session_state)

st.title("👥 Employee Management")
st.markdown("Manage employee records, departments, and onboarding.")

def reset_employee_editor_state(clear_selection=False):
    st.session_state.pop("employee_delete_confirm", None)
    if clear_selection:
        st.session_state.pop("employee_management_selected", None)

notice = st.session_state.pop("employee_management_notice", None)
if notice:
    st.toast(notice["message"])

info_message = st.session_state.pop("employee_management_info", None)
if info_message:
    st.toast(info_message)

employees = get_all_employees()
employee_df = get_employee_dataframe()
display_employee_df = employee_df.drop(columns=["id"], errors="ignore").rename(
    columns={
        "code": "Employee Code",
        "name": "Name",
        "department": "Department",
    }
)

total_employees = len(display_employee_df.index)
total_departments = (
    int(display_employee_df["Department"].nunique())
    if not display_employee_df.empty
    else 0
)

metric_col_1, metric_col_2, _ = st.columns([1, 1, 2])
with metric_col_1:
    st.metric("Total Employees", total_employees)
with metric_col_2:
    st.metric("Departments", total_departments)

st.divider()

# Create tabs for better space management
tab_list, tab_add, tab_edit = st.tabs(["📋 Directory", "➕ Add / Import", "✏️ Edit / Delete"])

# TAB 1: Directory
with tab_list:
    search_query = st.text_input(
        "Search by name, code, or department",
        placeholder="Type to search...",
    )

    if display_employee_df.empty:
        st.info("No employee data is available yet.")
    else:
        if search_query:
            mask = display_employee_df.apply(
                lambda row: row.astype(str).str.contains(search_query, case=False).any(),
                axis=1,
            )
            filtered_df = display_employee_df[mask]
        else:
            filtered_df = display_employee_df

        if filtered_df.empty:
            st.warning("No matching employees were found.")
        else:
            st.dataframe(filtered_df, use_container_width=True, hide_index=True, height=400)


# TAB 2: Add/Import
with tab_add:
    col_add, col_import = st.columns(2, gap="large")
    
    with col_add:
        with st.container(border=True):
            st.subheader("Add Single Employee")
            with st.form("add_employee_form", clear_on_submit=True):
                # Compact form using columns
                c1, c2 = st.columns(2)
                with c1:
                    name = st.text_input("Full Name", placeholder="e.g. John Doe")
                with c2:
                    department = st.text_input("Department", placeholder="e.g. IT")
                
                add_employee_submitted = st.form_submit_button(
                    "Add Employee",
                    use_container_width=True,
                )

            if add_employee_submitted:
                errors = validate_employee_fields(name, department)
                if errors:
                    st.toast("⚠️ " + errors[0])
                else:
                    employee_code = create_employee(name, department)
                    st.toast(f"✅ Employee added with code {employee_code}.")
                    st.rerun()

    with col_import:
        with st.container(border=True):
            st.subheader("Bulk Import from Excel")
            st.caption("Upload an Excel file with 'Name' and 'Department' columns.")
            
            uploaded_file = st.file_uploader(
                "Choose Excel file",
                type=["xlsx", "xls"],
                label_visibility="collapsed",
            )

            if uploaded_file is not None and st.button(
                "Start Import",
                use_container_width=True,
            ):
                try:
                    import_df = pd.read_excel(uploaded_file)
                    if "Name" not in import_df.columns or "Department" not in import_df.columns:
                        st.toast("⚠️ Missing 'Name' or 'Department' columns in Excel.")
                    else:
                        imported_count = 0
                        for _, row in import_df.iterrows():
                            if pd.notna(row["Name"]) and pd.notna(row["Department"]):
                                create_employee(str(row["Name"]), str(row["Department"]))
                                imported_count += 1

                        st.session_state["employee_management_notice"] = {
                            "message": f"✅ Successfully imported {imported_count} employees.",
                        }
                        st.rerun()
                except Exception as error:
                    st.toast(f"⚠️ Error reading Excel: {error}")


# TAB 3: Edit/Delete
with tab_edit:
    if not employees:
        st.info("No employees available to edit.")
    else:
        employee_map = {
            employee_code: {
                "id": employee_id,
                "employee_code": employee_code,
                "name": name,
                "department": department,
            }
            for employee_id, employee_code, name, department in employees
        }

        with st.container(border=True):
            selected_employee_code = st.selectbox(
                "Select employee to edit",
                list(employee_map.keys()),
                format_func=lambda code: f"{code} - {employee_map[code]['name']} ({employee_map[code]['department']})",
                key="employee_management_selected",
            )
            selected_employee = employee_map[selected_employee_code]

            with st.form("edit_employee_form"):
                # Compact form using columns
                c1, c2 = st.columns(2)
                with c1:
                    edit_name = st.text_input("Name", value=selected_employee["name"])
                with c2:
                    edit_department = st.text_input("Department", value=selected_employee["department"])
                    
                update_employee_submitted = st.form_submit_button(
                    "Update Employee",
                    use_container_width=True,
                )

            if update_employee_submitted:
                errors = validate_employee_fields(edit_name, edit_department)
                if errors:
                    st.toast("⚠️ " + errors[0])
                else:
                    update_employee(selected_employee["id"], edit_name, edit_department)
                    reset_employee_editor_state(clear_selection=False)
                    st.session_state["employee_management_notice"] = {
                        "message": f"✅ Updated employee {selected_employee['employee_code']}.",
                    }
                    st.rerun()

        with st.expander("⚠️ Danger Zone (Delete Employee)"):
            st.warning("Deleting an employee also removes that employee's saved face images. This action cannot be undone.")
            st.checkbox(
                "I understand and want to continue.",
                key="employee_delete_confirm",
            )

            if st.button("Delete Employee", type="primary"):
                if not st.session_state.get("employee_delete_confirm"):
                    st.toast("⚠️ Please check the confirmation box before deleting.")
                else:
                    deleted_code = delete_employee(selected_employee["id"])
                    reset_employee_editor_state(clear_selection=True)
                    st.session_state["employee_management_notice"] = {
                        "message": f"✅ Deleted employee {deleted_code}.",
                    }
                    st.rerun()
