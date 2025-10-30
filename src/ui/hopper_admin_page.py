# ui/hopper_admin_page.py
import streamlit as st
from data.db import Database


class HopperAdminPage:
    """
    Admin interface for managing material-to-hopper mappings.

    Provides functionality to view, update, and validate hopper assignments
    for each material. Ensures data integrity by preventing duplicate hopper
    assignments and enforcing valid input formats.
    """

    def __init__(self) -> None:
        """Initialize database connection and load materials and mappings."""
        self.db = Database()
        self.materials = self.db.materials
        self.material_hoppers = self.db.get_material_hoppers()

    # ---------------------------------------------------
    #  Form Handling
    # ---------------------------------------------------
    def render_form(self) -> None:
        """
        Renders the editable hopper mapping form.

        Users can update hopper numbers for each material.
        Validates input and prevents duplicate assignments before saving.
        """
        updated_hoppers = {}
        cols = st.columns(3)


        for idx, material in enumerate(self.materials):
            col = cols[idx % 3]

            hopper_values = self.material_hoppers.get(material, [])
            hopper_nos = [int(h.replace("HOPPER_", "").replace("_ACT", "")) for h in hopper_values]
            default_input = "" if hopper_nos == [0] else ", ".join(map(str, sorted(hopper_nos)))

            input_value = col.text_input(
                f"{material}",
                value=default_input,
                key=f"{material}_input",
                placeholder="e.g., 1, 2, 3",
            )
            updated_hoppers[material] = input_value.strip()

        submitted = st.form_submit_button("💾 Save Mappings")

        if submitted:
            self.handle_submission(updated_hoppers)

    # ----------------------------------------------------
    #  Submission Handling
    # ----------------------------------------------------
    def handle_submission(self, updated_hoppers) -> None:
        """
        Parses, validates, and updates hopper mappings.

        Args:
            updated_hoppers (dict): Material → comma-separated hopper numbers
        """
        used_hoppers = {}
        duplicate_errors = []
        final_updates = {}

        # Parse & validate
        for material, hopper_str in updated_hoppers.items():
            try:
                numbers = [
                    int(n.strip())
                    for n in hopper_str.split(",")
                    if n.strip().isdigit() and 1 <= int(n.strip()) <= 19
                ] or [0]

                for n in numbers:
                    if n == 0:
                        continue
                    if n in used_hoppers:
                        duplicate_errors.append(
                            f"Hopper {n} already assigned to '{used_hoppers[n]}' "
                            f"(conflict with '{material}')"
                        )
                    else:
                        used_hoppers[n] = material

                final_updates[material] = numbers

            except Exception as e:
                st.error(f"Error parsing {material}: {e}")
                return

        if duplicate_errors:
            st.warning("⚠ Duplicate hopper numbers detected:")
            for err in duplicate_errors:
                st.write(f"- {err}")
            return

        # Apply updates
        errors = False
        for material, numbers in final_updates.items():
            try:
                self.db.update_material_hoppers(material, numbers)
            except Exception as e:
                st.error(f"Error updating {material}: {e}")
                errors = True

        if not errors:
            st.session_state["hopper_success_message"] = "✅ Hopper mappings updated successfully."
            st.session_state["navigation_target"] = "welcome"
            st.rerun()

    # ----------------------------------------------------
    #  Render Page
    # ----------------------------------------------------
    def render(self) -> None:
        """
        Renders the Hopper Admin Page.

        Displays the form for updating hopper mappings as well as
        the read-only current mappings.
        """
        st.subheader("Hopper Material Mapping")
        # st.info("Enter hopper numbers from 1 to 19 ")

        # Show any success message from previous submission
        if st.session_state.get("hopper_success_message"):
            st.success(st.session_state["hopper_success_message"])
            st.session_state.pop("hopper_success_message", None)

        # Render form inside Streamlit form
        with st.form(key="hopper_map_form", clear_on_submit=False):
            self.render_form()

        # Display current mappings outside the form
        st.markdown("### 📋 Current Material → Hopper Mapping")
        st.write(self.db.get_material_hoppers())


# -------------------------------
# Entry Point
# -------------------------------
def hopper_admin_page() -> None:
    """
    Streamlit entry point for Hopper Admin Page.
    Instantiates and renders the HopperAdminPage class.
    """
    HopperAdminPage().render()
