import streamlit as st

from src.rag import answer_question


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Clinical Guideline RAG",
    page_icon="🩺",
    layout="wide",
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🩺 Clinical Guideline Assistant")

st.markdown(
    """
    **RAG system for retrieving evidence from official
    clinical guidelines.**
    
    The system runs locally using:
    - Qdrant
    - Ollama
    - Qwen3
    - Local embeddings
    """
)


st.divider()


# --------------------------------------------------
# Question Input
# --------------------------------------------------

question = st.text_area(
    "Ask a question",
    placeholder=(
        "Example: What is the recommended blood pressure target?"
    ),
    height=120,
)


# --------------------------------------------------
# Search Button
# --------------------------------------------------

if st.button(
    "🔍 Search Guidelines",
    type="primary",
    use_container_width=True,
):

    if not question.strip():

        st.warning(
            "Please enter a question first."
        )

    else:

        with st.spinner(
            "Searching clinical guidelines..."
        ):

            try:

                result = answer_question(
                    question.strip()
                )

            except Exception as e:

                st.error(
                    "An error occurred while running the RAG pipeline."
                )

                st.exception(e)

                st.stop()


        # --------------------------------------------------
        # Status
        # --------------------------------------------------

        if result["status"] == "rejected":

            st.error(
                result["recommendation"]
            )

            st.info(
                result["disclaimer"]
            )

            st.stop()


        # --------------------------------------------------
        # Recommendation
        # --------------------------------------------------

        st.subheader("💡 Recommendation")

        st.write(
            result["recommendation"]
        )


        # --------------------------------------------------
        # Confidence
        # --------------------------------------------------

        st.subheader("📊 Confidence")

        confidence = result["confidence"]

        if confidence == "HIGH":

            st.success(
                "🟢 HIGH"
            )

        elif confidence == "MEDIUM":

            st.warning(
                "🟡 MEDIUM"
            )

        else:

            st.error(
                "🔴 LOW"
            )


        # --------------------------------------------------
        # Supporting Evidence
        # --------------------------------------------------

        st.subheader(
            "📚 Supporting Evidence"
        )

        evidence = result.get(
            "supporting_evidence",
            []
        )

        if evidence:

            for i, item in enumerate(
                evidence,
                start=1
            ):

                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"**Evidence {i}**"
                    )

                    st.write(item)

        else:

            st.info(
                "No supporting evidence was returned."
            )


        # --------------------------------------------------
        # Citations
        # --------------------------------------------------

        st.subheader(
            "📖 Sources"
        )

        citations = result.get(
            "citations",
            []
        )

        if citations:

            for i, citation in enumerate(
                citations,
                start=1
            ):

                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"### Source {i}"
                    )

                    st.write(
                        f"**Document:** "
                        f"{citation.get('document', 'Unknown')}"
                    )

                    st.write(
                        f"**Organization:** "
                        f"{citation.get('organization', 'Unknown')}"
                    )

                    st.write(
                        f"**Section:** "
                        f"{citation.get('section', 'Unknown')}"
                    )

                    st.write(
                        f"**Page:** "
                        f"{citation.get('page', 'Unknown')}"
                    )

                    st.code(
                        citation.get(
                            "chunk_id",
                            "Unknown"
                        )
                    )

        else:

            st.info(
                "No citations were returned."
            )


        # --------------------------------------------------
        # Disclaimer
        # --------------------------------------------------

        st.divider()

        st.caption(
            result["disclaimer"]
        )