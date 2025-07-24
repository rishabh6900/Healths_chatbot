import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load API key from .env
load_dotenv()

# Initialize the model
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

# Prompt for disease information
disease_prompt = PromptTemplate(
    template="Provide detailed information about {disease}, including symptoms, causes, and risk factors.",
    input_variables=["disease"]
)

# Prompt for treatment 
treatment_prompt = PromptTemplate(
    template="""
    Based on the following disease information: {disease_info}

    1. Suggest appropriate medications (generic and brand names if available)
    2. Explain how each medication works to treat the disease
    3. Provide dosage guidelines and potential side effects
    4. Mention any important precautions or contraindications
    """,
    input_variables=["disease_info"]
)

# chains
disease_chain = disease_prompt | model | StrOutputParser()
treatment_chain = treatment_prompt | model | StrOutputParser()

# Combine chains
full_chain = {
    "disease_info": disease_chain,
    "disease": RunnablePassthrough()
} | treatment_chain

# Streamlit app UI
st.title("🩺 Disease Diagnosis and Treatment Assistant")

disease_input = st.text_input("Enter a disease name:", placeholder="e.g., Diabetes")

if st.button("Get Treatment Plan") and disease_input:
    with st.spinner("Analyzing..."):
        result = full_chain.invoke({"disease": disease_input})
        st.subheader("Recommended Treatment:")
        st.markdown(result)
