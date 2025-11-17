import streamlit as st
import json
import re
import os
from typing import List, Dict, Any, Optional, Union

# Path to the JSON file
JSON_PATH = "data/W02/R01_create_ema_registries_with_publis/test/registries_dataset_enhanced_v3.json"


def load_data():
    """Load the registry data from JSON file."""
    try:
        with open(JSON_PATH, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return []


def clean_text(text: str) -> str:
    """Remove punctuation and convert to lowercase."""
    if not text:
        return ""
    return re.sub(r"[^\w\s]", "", text.lower())


def format_list_value(value: Union[List, str, None]) -> str:
    """Format a list value or return 'Not Found' if empty."""
    if value is None or value == "" or (isinstance(value, list) and len(value) == 0):
        return "Not Found"

    if isinstance(value, list):
        if len(value) == 1:
            return value[0]
        return ", ".join(value)

    return str(value)


def search_registries(data: List[Dict], query: str) -> List[Dict]:
    """Search for registries matching the query."""
    if not query:
        return []

    query_terms = clean_text(query).split()
    results = []

    for registry in data:
        registry_name_clean = clean_text(registry["registry_name"])
        match = all(term in registry_name_clean for term in query_terms)

        if match:
            results.append(registry)

    return results


def main():
    st.set_page_config(page_title="Registry Explorer", layout="wide")

    # Load data
    registries_data = load_data()

    # Sidebar for search
    st.sidebar.markdown(
        """
        <style>
        .sidebar-label {
            font-size: 24px !important;
            font-weight: bold;
        }
        .sidebar-textarea textarea {
            font-size: 20px !important;
            min-height: 80px !important;
            max-height: 120px !important;
            line-height: 1.3 !important;
            resize: vertical !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<div class="sidebar-label">Enter registry name:</div>', unsafe_allow_html=True
    )
    query = st.sidebar.text_area(
        " ",
        key="query_box",
        height=80,
        help="Type your query here",
        placeholder="Type registry name...",
        label_visibility="collapsed",
    )

    # Main content
    st.title("Registry Database Explorer")

    # Section 1: Search Results
    if query:
        matching_registries = search_registries(registries_data, query)

        if not matching_registries:
            st.header(f"Found 0 matching registries")
            st.warning("No registry found")
        else:
            st.header(f"Found {len(matching_registries)} matching registries")

            # Prepare data for display
            display_data = []
            for reg in matching_registries:
                display_data.append(
                    {
                        "Registry Name": reg["registry_name"],
                        "Acronym": reg.get("acronym", "Not Found"),
                        "Geographical Area": format_list_value(
                            reg.get("geographical_area")
                        ),
                        "N\u00b0 of publis": reg.get(
                            "number_of_occurrences", "Not Found"
                        ),
                    }
                )

            # Custom CSS for larger font in the table and selectbox
            st.markdown(
                """
                <style>
                .big-table td, .big-table th {
                    font-size: 24px !important;
                }
                .stSelectbox label, .stSelectbox div[data-baseweb="select"] {
                    font-size: 24px !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Render table with larger font using HTML
            table_html = "<table class='big-table' style='width:100%; border-collapse:collapse;'>"
            table_html += (
                "<tr>"
                + "".join(f"<th>{col}</th>" for col in display_data[0].keys())
                + "</tr>"
            )
            for row in display_data:
                table_html += (
                    "<tr>"
                    + "".join(f"<td>{val}</td>" for val in row.values())
                    + "</tr>"
                )
            table_html += "</table>"
            st.markdown(table_html, unsafe_allow_html=True)

            # Selection for showing publications
            if matching_registries:
                st.markdown(
                    """
                    <style>
                    .select-registry-label {
                        font-size: 28px !important;
                        font-weight: bold;
                        margin-bottom: 16px !important;
                    }
                    .big-selectbox .stSelectbox div[data-baseweb="select"] {
                        min-height: 60px !important;
                        font-size: 24px !important;
                        padding-top: 12px !important;
                        padding-bottom: 12px !important;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(
                    '<div class="select-registry-label">Select a registry to show publications:</div>',
                    unsafe_allow_html=True,
                )
                with st.container():
                    selected_registry_name = st.selectbox(
                        "Select a registry to show publications:",
                        [reg["registry_name"] for reg in matching_registries],
                        key="select_registry_box",
                        label_visibility="collapsed",
                    )
                # Custom styled Show Publications button
                st.markdown(
                    """
                    <style>
                    .red-outline-btn button {
                        border: 3px solid #d7263d !important;
                        color: #d7263d !important;
                        font-size: 28px !important;
                        padding: 1.2em 2.5em !important;
                        border-radius: 12px !important;
                        font-weight: bold !important;
                        background: transparent !important;
                        min-width: 300px !important;
                        min-height: 60px !important;
                    }
                    .red-outline-btn button:hover {
                        background: #ffeaea !important;
                        color: #a81c2a !important;
                    }
                    </style>
                    <div class='red-outline-btn'>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                show_publications = st.button("Show Publications", key="show_publi_btn")

                # Section 2: Registry Publication Details
                if show_publications or st.session_state.get(
                    "show_publications_active"
                ):
                    st.session_state["show_publications_active"] = True
                    st.header("Registry Publication Details")

                    # Find the selected registry
                    selected_registry = next(
                        (
                            reg
                            for reg in matching_registries
                            if reg["registry_name"] == selected_registry_name
                        ),
                        None,
                    )

                    if selected_registry and selected_registry.get("list_publi_ids"):
                        publications = selected_registry["list_publi_ids"]

                        if not publications:
                            st.info(
                                f"No publications found for {selected_registry_name}"
                            )
                        else:
                            st.subheader(
                                f"Found {len(publications)} publications for {selected_registry_name}"
                            )

                            # --- Pagination setup ---
                            PAGE_SIZE = 30
                            if "publi_page" not in st.session_state:
                                st.session_state.publi_page = 0
                            # Publications to show (filtered or not)
                            if "filtered_publications" not in st.session_state:
                                st.session_state.filtered_publications = None
                            if "selected_publications" not in st.session_state:
                                st.session_state.selected_publications = set()
                            if "show_abstract" not in st.session_state:
                                st.session_state.show_abstract = None

                            publications_to_show = (
                                st.session_state.filtered_publications or publications
                            )
                            total_pages = (
                                len(publications_to_show) - 1
                            ) // PAGE_SIZE + 1
                            # Clamp publi_page to valid range
                            st.session_state.publi_page = max(
                                0, min(st.session_state.publi_page, total_pages - 1)
                            )
                            start_idx = st.session_state.publi_page * PAGE_SIZE
                            end_idx = start_idx + PAGE_SIZE
                            page_publications = publications_to_show[start_idx:end_idx]

                            # --- Filtering logic ---
                            col1, col2, col3 = st.columns([1, 1, 6])
                            with col1:
                                filter_btn = st.button(
                                    "Filter on selected",
                                    disabled=len(st.session_state.selected_publications)
                                    == 0,
                                    type=(
                                        "primary"
                                        if len(st.session_state.selected_publications)
                                        > 0
                                        else "secondary"
                                    ),
                                    help="Filter table to only show selected publications",
                                )
                            with col2:
                                reset_btn = st.button(
                                    "Remove Filter",
                                    disabled=st.session_state.filtered_publications
                                    is None,
                                )
                            with col3:
                                prev_btn, next_btn = st.columns([1, 1])
                                with prev_btn:
                                    if st.button(
                                        "Previous Page",
                                        disabled=st.session_state.publi_page == 0,
                                    ):
                                        st.session_state.publi_page = max(
                                            0, st.session_state.publi_page - 1
                                        )
                                        st.experimental_rerun()
                                with next_btn:
                                    if st.button(
                                        "Next Page",
                                        disabled=st.session_state.publi_page
                                        >= total_pages - 1,
                                    ):
                                        st.session_state.publi_page = min(
                                            total_pages - 1,
                                            st.session_state.publi_page + 1,
                                        )
                                        st.experimental_rerun()

                            # Apply filter when button is clicked
                            if (
                                filter_btn
                                and len(st.session_state.selected_publications) > 0
                            ):
                                st.session_state.filtered_publications = [
                                    p
                                    for p in publications
                                    if p.get("publication_id")
                                    in st.session_state.selected_publications
                                ]
                                st.session_state.publi_page = 0
                                st.experimental_rerun()

                            # Reset filter
                            if reset_btn:
                                st.session_state.filtered_publications = None
                                st.session_state.selected_publications = set()
                                st.session_state.publi_page = 0
                                st.experimental_rerun()

                            # --- Table column headers ---
                            table_columns = [
                                "Select",
                                "Index",
                                "Abstract",
                                "Title",
                                "Medical Condition",
                                "Outcome Measure",
                                "Geographical Area",
                                "Population Size",
                                "Population Follow Up",
                                "Population Age Group",
                                "Population Sex",
                            ]
                            st.markdown(
                                """
                                <style>
                                .not-found {
                                    color: #888;
                                    font-style: italic;
                                    font-size: 0.9em;
                                }
                                .publi-table th {
                                    position: sticky;
                                    top: 0;
                                    background: linear-gradient(90deg, #222 0%, #444 100%); /* dark */
                                    color: #fff;
                                }
                                @media (prefers-color-scheme: light) {
                                    .publi-table th {
                                        background: linear-gradient(90deg, #e0e0e0 0%, #f8f8f8 100%);
                                        color: #222;
                                    }
                                }
                                .publi-table {
                                    border-collapse: collapse;
                                    width: 100%;
                                }
                                .publi-table th, .publi-table td {
                                    border: 1px solid #ddd;
                                    padding: 8px;
                                }
                                /* Modal styles */
                                .modal-bg {
                                    position: fixed;
                                    top: 0; left: 0; right: 0; bottom: 0;
                                    background: rgba(0,0,0,0.5);
                                    z-index: 9999;
                                    display: flex;
                                    align-items: center;
                                    justify-content: center;
                                }
                                .modal-content {
                                    background: #fff;
                                    padding: 2em 2.5em;
                                    border-radius: 16px;
                                    box-shadow: 0 4px 32px rgba(0,0,0,0.25);
                                    max-width: 600px;
                                    min-width: 350px;
                                    text-align: center;
                                }
                                .modal-title {
                                    font-size: 2em;
                                    font-weight: bold;
                                    margin-bottom: 1em;
                                }
                                .modal-abstract {
                                    font-size: 1.2em;
                                    margin-bottom: 2em;
                                }
                                .modal-close-btn {
                                    font-size: 1.1em;
                                    padding: 0.5em 2em;
                                    border-radius: 8px;
                                    border: 2px solid #d7263d;
                                    color: #d7263d;
                                    background: #fff;
                                    font-weight: bold;
                                    cursor: pointer;
                                }
                                .modal-close-btn:hover {
                                    background: #ffeaea;
                                }
                                </style>
                                """,
                                unsafe_allow_html=True,
                            )
                            # --- Table rendering ---
                            table_html = "<table class='publi-table'>"
                            table_html += (
                                "<tr>"
                                + "".join(f"<th>{col}</th>" for col in table_columns)
                                + "</tr>"
                            )
                            for i, publication in enumerate(page_publications):
                                publi_id = publication.get(
                                    "publication_id", f"unknown_{i}"
                                )
                                is_selected = (
                                    publi_id in st.session_state.selected_publications
                                )

                                def nf(val):
                                    if (
                                        val is None
                                        or val == ""
                                        or val == "Not specified"
                                        or val == "Not found"
                                    ):
                                        return (
                                            "<span class='not-found'>Not found</span>"
                                        )
                                    if isinstance(val, list):
                                        if not val:
                                            return "<span class='not-found'>Not found</span>"
                                        return ", ".join(
                                            (
                                                f"<span class='not-found'>Not found</span>"
                                                if v == "Not specified"
                                                or v == "Not found"
                                                else str(v)
                                            )
                                            for v in val
                                        )
                                    return (
                                        f"<span class='not-found'>Not found</span>"
                                        if val == "Not specified" or val == "Not found"
                                        else str(val)
                                    )

                                table_html += f"<tr>"
                                # Select checkbox
                                checked = "checked" if is_selected else ""
                                table_html += f"<td><input type='checkbox' {checked} onclick=\"window.parent.postMessage({{type: 'select_publi', publi_id: '{publi_id}', checked: this.checked}}, '*')\"></td>"
                                # Index column
                                table_html += f"<td>{start_idx + i + 1}</td>"
                                # Show Abstract button
                                table_html += f"<td><button onclick=\"window.parent.postMessage({{type: 'show_abstract', publi_id: '{publi_id}'}}, '*')\">Show Abstract</button></td>"
                                table_html += f"<td>{nf(publication.get('title'))}</td>"
                                table_html += f"<td>{nf(publication.get('medical_condition'))}</td>"
                                table_html += (
                                    f"<td>{nf(publication.get('outcome_measure'))}</td>"
                                )
                                table_html += f"<td>{nf(publication.get('geographical_area'))}</td>"
                                table_html += (
                                    f"<td>{nf(publication.get('population_size'))}</td>"
                                )
                                table_html += f"<td>{nf(publication.get('population_follow_up'))}</td>"
                                table_html += f"<td>{nf(publication.get('population_age_group'))}</td>"
                                table_html += (
                                    f"<td>{nf(publication.get('population_sex'))}</td>"
                                )
                                table_html += "</tr>"
                            table_html += "</table>"
                            st.markdown(table_html, unsafe_allow_html=True)

                            # --- JS for checkbox and button events ---
                            st.components.v1.html(
                                f"""
                                <script>
                                window.addEventListener('message', (event) => {{
                                    if (event.data.type === 'select_publi') {{
                                        fetch('/_stcore/select_publi', {{
                                            method: 'POST',
                                            headers: {{'Content-Type': 'application/json'}},
                                            body: JSON.stringify({{publi_id: event.data.publi_id, checked: event.data.checked}})
                                        }}).then(() => window.location.reload());
                                    }}
                                    if (event.data.type === 'show_abstract') {{
                                        window.localStorage.setItem('show_abstract_publi_id', event.data.publi_id);
                                        window.location.reload();
                                    }}
                                }});
                                </script>
                                """,
                                height=0,
                            )

                            # --- Abstract modal ---
                            show_abstract_publi_id = st.session_state.get(
                                "show_abstract_publi_id", None
                            )
                            if show_abstract_publi_id or st.session_state.get(
                                "show_abstract"
                            ):
                                publi_id = (
                                    show_abstract_publi_id
                                    or st.session_state.get("show_abstract")
                                )
                                publication = next(
                                    (
                                        p
                                        for p in publications
                                        if p.get("publication_id") == publi_id
                                    ),
                                    None,
                                )
                                if publication:
                                    st.markdown(
                                        f"""
                                        <div class='modal-bg'>
                                            <div class='modal-content'>
                                                <div class='modal-title'>{publication.get('title', 'No Title')}</div>
                                                <div class='modal-abstract'>{publication.get('abstract', 'No abstract available')}</div>
                                                <button class='modal-close-btn' onclick="window.localStorage.removeItem('show_abstract_publi_id'); window.location.reload();">Close</button>
                                            </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True,
                                    )
                    else:
                        st.info(f"No publications found for {selected_registry_name}")


if __name__ == "__main__":
    main()
