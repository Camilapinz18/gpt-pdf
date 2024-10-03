import os
import uuid
from typing import Any
import requests
import urllib.parse
import json
from pyproj import Proj, Transformer
import pandas as pd
import pickle
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.callbacks import get_openai_callback
from PyPDF2 import PdfReader
from streamlit_extras.colored_header import colored_header
from dotenv import load_dotenv



# load the environment variables
load_dotenv()


# Create the vectorstore directory if it doesn't exist
VECTORSTORE_DIR = "vectorstore"
os.makedirs(VECTORSTORE_DIR, exist_ok=True)


# set the page title and icon
st.set_page_config(page_title="Curadur-IA San Isidro", page_icon=":house:")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def save_vectorstore(vectorstore, filename):
    with open(os.path.join(VECTORSTORE_DIR, filename), "wb") as file:
        pickle.dump(vectorstore, file)


def get_vectorstore(text_chunks, embeddings_selection):
    if embeddings_selection == "OpenAI":
        embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY
        )
    else:
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def load_vectorstore(filename):
    with open(os.path.join(VECTORSTORE_DIR, filename), "rb") as file:
        vectorstore = pickle.load(file)
    return vectorstore


if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )


def chain_setup(vectorstore, model_name="OpenAI"):
    template = """{question}
    """

    if model_name == "OpenAI":
        # initialize the LLM with api key
        llm = ChatOpenAI(
            openai_api_key=OPENAI_API_KEY
        )

    elif model_name == "Falcon":
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-xxl",
            model_kwargs={"temperature": 0.5, "max_length": 512},
        )

    elif model_name == "OpenAssistant":
        template = """<|prompter|>{question}<|endoftext|>
        <|assistant|>"""
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm = HuggingFaceHub(
            repo_id="OpenAssistant/oasst-sft-4-pythia-12b-epoch-3.5",
            model_kwargs={"max_new_tokens": 1200},
        )

    else:
        raise ValueError(
            "Invalid model_name. Choose from 'OpenAI', 'Falcon', 'OpenAssistant'."
        )

    # Use the memory from the session state
    memory = st.session_state.memory

    # Create the LLM chain
    if model_name == "OpenAI":
        llm_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=vectorstore.as_retriever(), memory=memory
        )
    else:
        llm_chain = LLMChain(llm=llm, prompt=prompt, memory=memory)
    return llm_chain


# generate response
def generate_response(question, llm_chain, llm_model_name):
    cost = 0.0

    # Get the response from the LLM
    if llm_model_name == "OpenAI":
        with get_openai_callback() as cb:
            response = llm_chain.run(question)

            if cb is not None:
                cost = round(cb.total_cost, 5)
    else:
        response = llm_chain.run(question)

    st.session_state.chat_history.append((question, response))
    return response, cost


# This function will create a new message in the chat
def render_message(sender: str, avatar_url: str, message: str, cost: float = 0.0):
    # Create a container for the message
    with st.container():
        # Create a column for the avatar
        col1, col2 = st.columns([1, 9])

        # Display the avatar
        with col1:
            st.image(avatar_url, width=50)
            if sender == "AI":
                st.write(cost)

        # Display the message
        with col2:
            # st.markdown(f"**{sender}**")
            if sender == "User":
                st.text_area("", value=message, height=50, max_chars=None, key=None)
            else:
                st.info(message)


# This function will get current vectorstore
def get_current_vectorstore():
    if st.session_state.vectorstore_selection == "Create New":
        st.warning(
            "Please upload PDFs to create a new vectorstore or select an existing vectorstore."
        )
        return None
    else:
        vectorstore = load_vectorstore(st.session_state.vectorstore_selection)
        return vectorstore


def search_address(
    street,
    number,
    municipality,
    province,
    country
) -> Any:


    street_name = street
    street_number = number
    municipality = municipality
    provincia = province
    country = country
    address = f'{street_name} {street_number}, {municipality}, Provincia de {provincia}, {country}'
    encoded_address = urllib.parse.quote(address)

    url_base = f'https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates?SingleLine={encoded_address}&f=json&outFields=*'
    response = requests.get(url_base)

    if response.status_code == 200:
        data = response.json()
        print("*"*50)
        print("data1", data)

        if "candidates" in data and len(data["candidates"]) > 0:
            coordenates = data["candidates"][0].get("location", "No se encontró el campo 'coordenates'")
            print("coordenates", coordenates)
        else:
            print("status_code=404, detail=Coordenadas no identificadas")

    #COnvertir coordenadas a Webmercaotr:
    transformer = Transformer.from_crs("epsg:4326", "epsg:102100", always_xy=True)
    lat, lon = coordenates['y'], coordenates['x']
    x, y = transformer.transform(lon, lat)
    print(" X,Y>", x,y)

    url = "https://msi-gis.gestionmsi.gob.ar/server/rest/services/PARCELAS20/FeatureServer/1/query"

    geometry = {
        "x": x,
        "y": y,
        "spatialReference": {"wkid": 102100}
    }

    geometry_json = json.dumps(geometry)

    params = {
        'f': 'json',
        'returnGeometry': 'true',
        'spatialRel': 'esriSpatialRelIntersects',
        'geometryType': 'esriGeometryPoint',
        'inSR': '102100',
        'outFields': '*',
        'outSR': '102100',
        'resultType': 'tile'
    }

    encoded_params = urllib.parse.urlencode(params)
    full_url = f"{url}?{encoded_params}&geometry={urllib.parse.quote(geometry_json)}"

    response = requests.get(full_url)
    if response.status_code == 200:
        data = response.json()
        print("/"*50)
        print("data2>", data)

        data_return = {
            "address": address,
            "nomenclature": data['features'][0]['attributes']['NOMEN'],
            "circumscription": data['features'][0]['attributes']['CIR'],
            "sector": data['features'][0]['attributes']['SEC'],
            "block": data['features'][0]['attributes']['MAN'],
            "plot": data['features'][0]['attributes']['PAR'],
            "zone": data['features'][0]['attributes']['ZONA1'],
            "radius": data['features'][0]['attributes']['RADIO1'],
            "area": data['features'][0]['attributes']['Shape__Area'] * data['features'][0]['attributes']['Shape__Length'],
            "full_address": f"{address}, NOMEN: {data['features'][0]['attributes']['NOMEN']}, CIR: {data['features'][0]['attributes']['CIR']}, SEC: {data['features'][0]['attributes']['SEC']}, MAN: {data['features'][0]['attributes']['MAN']}, PAR: {data['features'][0]['attributes']['PAR']}, ZONA: {data['features'][0]['attributes']['ZONA1']}, RADIO: {data['features'][0]['attributes']['RADIO1']}"
        }

        return data_return

    else:
        print("status_code=404, detail=Error encontrando el predio")

def main():
    page_bg_color = """
    <style>
        body {
            background-color: #FF0000;
        }
    </style>
    """
    st.markdown(page_bg_color, unsafe_allow_html=True)
    st.header("Curaduría San Isidro")

    with st.sidebar:
        st.subheader(":gear: Options")
        
        # Step 1: Choose a Large Language Model
        llm_selection = st.selectbox(
            "Step 1: Choose a Large Language Model",
            options=["OpenAI", "Falcon", "OpenAssistant"]
        )

        # Step 2: Choose Embeddings Model
        embeddings_selection = st.selectbox(
            "Step 2: Choose an Embeddings Model",
            options=["OpenAI", "HuggingFaceInstruct"]
        )

        # Step 3: Select or Create a Vector Store File
        vectorstore_files = ["Create New"] + os.listdir(VECTORSTORE_DIR)
        st.session_state.vectorstore_selection = st.selectbox(
            "Step 3: Choose a Vector Store File",
            options=vectorstore_files
        )

        # Handle file upload
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            type=["pdf", "txt"],
            accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                # Get PDF text
                raw_text = get_pdf_text(pdf_docs)

                # Get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # Create or load vector store
                if (
                    st.session_state.vectorstore_selection == "Create New"
                    or not os.path.exists(
                        os.path.join(VECTORSTORE_DIR, st.session_state.vectorstore_selection)
                    )
                ):
                    vectorstore = get_vectorstore(text_chunks, embeddings_selection)
                    vectorstore_filename = f"{llm_selection}_{embeddings_selection}_{len(os.listdir(VECTORSTORE_DIR))}.pkl"
                    save_vectorstore(vectorstore, vectorstore_filename)
                    st.session_state.vectorstore_selection = vectorstore_filename  # Update the current selection to the new file
                else:
                    vectorstore = load_vectorstore(st.session_state.vectorstore_selection)
                    vectorstore.update(text_chunks)

                # Get the current vectorstore
                current_vectorstore = vectorstore

                # Create conversation chain
                if current_vectorstore is not None:
                    st.session_state.conversation = chain_setup(current_vectorstore, llm_selection)

        if st.button("Clear Chat"):
            st.session_state.user = []
            st.session_state.generated = []
            st.session_state.cost = []


    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Ubicación y uso", "Alturas edificación", "Mitigations", "DREAD", "Test Cases"])

    # Generate empty lists for generated and user.
    # Assistant Response
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I'm Assistant, \n \n How may I help you?"]

    # user question
    if "user" not in st.session_state:
        st.session_state["user"] = ["Hi!"]

    # chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Generate empty list for cost history
    if "cost" not in st.session_state:
        st.session_state["cost"] = [0.0]

    # Pestaña 1: Ingresar dirección
    with tab1:
        # st.markdown("Ingrese el uso que le dará a la parcela:")
        # zona_seleccionada = st.selectbox(
        #     'Selecciona una zona:',
        #     ['Vivienda multifamiliar',
        #      'Vivienda unifamiliar',
        #      'Comercio grupos I y II',
        #      'Servicios grupos I y II',
        #      'Comercio minorista I y II',
        #      'Servicios comerciales I y II',
        #      'Servicios al automotor II y III ',
        #      'Comercio selectivo',
        #      'Industria – Depósitos - Talleres',
        #      'Talleres y depósitos clase 5 y 6 ',
        #      'Industria existente en fracciones de 2000m2 y habilitada ',
        #      'Industria Náutica Deportiva', 
        #      'Recreativo (Clubes deportivos) ',
        #      'Esparcimiento Público ',
        #      'Parque de la Ribera y Esparcimiento público y semipúblico actividad náutico'
        #     ]
        # )


        st.markdown("Ingrese la dirección de la parcela:")

        with st.form(key='address_form'):
            st.header("Dirección:")
            street = st.text_input("Calle")
            number = st.text_input("Número")
            municipality = st.text_input("Municipio", value="San Isidro", disabled=True)
            province = st.text_input("Provincia", value="Buenos Aires", disabled=True)
            country = st.text_input("País", value="ARG", disabled=True)
            search_button = st.form_submit_button(label='Buscar')

        # Procesamiento de la dirección
        if search_button:
            if not street or not number:
                st.error("Por favor, complete todos los campos obligatorios: Calle y Número.")
            else:
                address_final = search_address(street, number, municipality, province, country)
                headers = {
                    "address": "Dirección",
                    "nomenclature": "Nomenclatura",
                    "circumscription": "Circunscripción",
                    "sector": "Sector",
                    "block": "Bloque",
                    "plot": "Parcela",
                    "zone": "Zona",
                    "radius": "Radio",
                    "area": "Área (m²)",
                    "full_address": "Dirección completa"
                }

                # Mostrar información de la parcela
                df = pd.DataFrame([(headers[key], value) for key, value in address_final.items()], columns=['Datos parcela', 'Valor'])
                df.loc[df['Datos parcela'] == 'Área (m²)', 'Valor'] = df.loc[df['Datos parcela'] == 'Área (m²)', 'Valor'].apply(lambda x: f"{x:,.2f}")
                st.table(df)

                # Generar y mostrar información de GPT
                user_input = (
                    f"Te voy a brindar una zona y un radio: Zona {address_final['zone']}, Radio {address_final['radius']}."
                  #  f"El uso de parcela seleccionado por el usuario es: {zona_seleccionada}."
                    " Por favor, proporciona la siguiente información en una lista numerada de ítems claros y concisos:"
                    "\n\n1. Usos permitidos predominantes."
                    "\n2. Usos permitidos complementarios."
                    "\n3. Densidades."
                    "\n4. Parcelamiento."
                    "\n5. Factor de ocupación de suelo (FOS)."
                    "\n6. Factor de ocupación total (FOT)."
                    "\n7. Retiros de frente."
                    "\n8. Retiros laterales."
                    "\n9. Profundidad edificable."
                    "\n10. Alturas máximas."
                    "\n11. Plano límite."
                    "\n12. Número de viviendas por parcela."
                    "\n13. Separaciones entre edificios."
                    "\n14. Usos diferenciados."
                    "\n15Proporciona cualquier otra información relevante sobre la zona y radio especificados."
                    #"\n\nFinalmente, verifica si el uso de parcela seleccionado coincide con los usos permitidos predominantes y proporciona una respuesta a modo de conlusión"
                )
                # Llamar a la API de OpenAI y mostrar la respuesta formateada
                if user_input:
                    current_vectorstore = load_vectorstore(st.session_state.vectorstore_selection)
                    if current_vectorstore is None:
                        return

                    llm_chain = chain_setup(current_vectorstore, llm_selection)
                    response, cost = generate_response(user_input, llm_chain, llm_selection)

                    # Procesar la respuesta para mostrarla como lista
                    response_list = [line.strip() for line in response.splitlines() if line.strip()]

                    # Mostrar la respuesta de OpenAI como un listado
                    st.markdown("### Normatividad:")
                    for item in response_list:
                        # Aplicar el icono solo si el ítem es un título principal (por ejemplo, empieza con un número)
                        if item[0].isdigit():
                            st.markdown(f"📌 {item}")
                        else:
                            st.markdown(f"- {item}")





if __name__ == "__main__":
    # Inicializar estado de la sesión
    main()
