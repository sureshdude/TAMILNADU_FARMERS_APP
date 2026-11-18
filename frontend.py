import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8001"

st.set_page_config(
    page_title="TN Farmers Scheme Assistant",
    page_icon="🌾",
    layout="wide",
)


def fetch_indexed_files() -> list[str]:
    """Fetch filenames already stored in the backend FAISS index."""
    try:
        res = requests.get(f"{BACKEND_URL}/indexed-files", timeout=10)
        if res.status_code == 200:
            return res.json().get("files", [])
    except requests.exceptions.RequestException:
        pass
    return []


# --- Sidebar: Multi-PDF Upload & Vector DB Controls ---
with st.sidebar:
    st.header("📄 Upload Scheme PDFs")
    st.caption("Select one or multiple Tamil Nadu Agriculture / Farmers Welfare scheme PDFs to index into FAISS.")

    uploaded_files = st.file_uploader(
        "Choose PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("📥 Upload & Index Selected PDFs", type="primary", use_container_width=True):
            existing_files = set(fetch_indexed_files())

            for pdf_file in uploaded_files:
                if pdf_file.name in existing_files:
                    st.info(f"Skipped **{pdf_file.name}** (already in FAISS index).")
                    continue

                with st.spinner(f"Indexing '{pdf_file.name}'..."):
                    try:
                        files = {
                            "file": (
                                pdf_file.name,
                                pdf_file.getvalue(),
                                "application/pdf",
                            )
                        }
                        res = requests.post(
                            f"{BACKEND_URL}/upload-pdf",
                            files=files,
                            timeout=300,
                        )
                        if res.status_code == 200:
                            data = res.json()
                            existing_files.add(pdf_file.name)
                            st.success(
                                f"Indexed **{data['filename']}** "
                                f"({data['pages_processed']} pages, {data['chunks_indexed']} chunks)."
                            )
                        else:
                            err_detail = res.json().get("detail", "Failed to index PDF.")
                            st.error(f"**{pdf_file.name}**: {err_detail}")
                    except requests.exceptions.ConnectionError:
                        st.error("Cannot reach FastAPI backend. Ensure Uvicorn is running on port 8001.")
                        break
                    except requests.exceptions.Timeout:
                        st.error(f"Timed out while indexing **{pdf_file.name}**.")

    # Show all documents currently persisted in FAISS
    st.divider()
    st.subheader("📚 Indexed Documents in FAISS")
    persisted_files = fetch_indexed_files()
    if persisted_files:
        for fname in persisted_files:
            st.write(f"✅ {fname}")
    else:
        st.caption("No PDFs indexed yet.")

    st.divider()
    if st.button("🗑️ Reset Vector DB", use_container_width=True):
        try:
            res = requests.delete(f"{BACKEND_URL}/reset-db", timeout=10)
            if res.status_code == 200:
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": "Vector database cleared! Upload one or more Tamil Nadu Farmers Scheme PDFs in the sidebar to begin.",
                        "sources": [],
                    }
                ]
                st.toast("FAISS index cleared!", icon="✅")
                st.rerun()
        except requests.exceptions.ConnectionError:
            st.error("Backend server is offline.")

# --- Main Area: Chatbot Interface ---
st.title("🌾 Tamil Nadu Farmers Scheme Chatbot")
st.caption("Ask questions across all your uploaded scheme PDFs about subsidies, crop loans, eligibility, and application procedures.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Vanakkam! Upload one or more Tamil Nadu Farmers Scheme PDFs in the sidebar, then ask me anything about eligibility, subsidy percentages, or how to apply.",
            "sources": [],
        }
    ]

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Referenced PDF Pages"):
                for src in msg["sources"]:
                    st.write(f"- {src}")

# Chat input
if user_prompt := st.chat_input("e.g., What is the subsidy percentage for drip irrigation under PMKSY in Tamil Nadu?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt, "sources": []})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching scheme documents..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/chat",
                    json={"question": user_prompt},
                    timeout=60,
                )
                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])

                    st.markdown(answer)
                    if sources:
                        with st.expander("📚 Referenced PDF Pages"):
                            for src in sources:
                                st.write(f"- {src}")

                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                else:
                    err_msg = response.json().get("detail", "Error retrieving answer.")
                    st.error(err_msg)
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to FastAPI backend at http://127.0.0.1:8001.")