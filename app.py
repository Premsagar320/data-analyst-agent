import tempfile
import streamlit as st
from agent import app

st.title("Data Analyst Agent")
st.caption("Upload a CSV and ask questions in plain English.")

uploaded = st.file_uploader("Upload a CSV", type="csv")
question = st.text_input("Ask a question about your data")

if uploaded and question and st.button("Analyze"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as f:
        f.write(uploaded.getvalue())
        path = f.name
    with st.spinner("Thinking..."):
        result = app.invoke({"csv_path": path, "question": question})
    st.subheader("Answer")
    st.write(result["answer"])
    st.caption(f"Retries used: {result['retries']}")
    with st.expander("Code the agent ran"):
        st.code(result["code"], language="python")
    if result["output"]:
        with st.expander("Raw output"):
            st.text(result["output"])
