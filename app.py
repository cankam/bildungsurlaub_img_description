import streamlit as st
import io
import sqlite3
import base64
from pydantic import BaseModel
from PIL import Image # Import von PIL.Image
import os
# from dotenv import load_dotenv, find_dotenv
# load_dotenv(find_dotenv(usecwd=True))

# model
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import SimpleJsonOutputParser

model = ChatGroq(model="meta-llama/llama-4-maverick-17b-128e-instruct")

#### database
DB_FILE = "buildings.db"

def setup_database():
    """
    Richtet die SQLite-Datenbanktabelle ein, falls sie nicht existiert.
    WICHTIG: Wenn Sie Änderungen an der Tabellenstruktur vornehmen (z.B. Spalten hinzufügen/umbenennen),
    müssen Sie die bestehende 'buildings.db'-Datei löschen, bevor Sie die App neu starten,
    damit die neue Tabellenstruktur angewendet wird.
    """
    db_conn = sqlite3.connect(DB_FILE)
    cursor = db_conn.cursor()
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS image_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name TEXT NOT NULL UNIQUE, 
            title TEXT NOT NULL,
            buildings TEXT,
            description TEXT,
            time_added DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    '''
    )
    db_conn.commit()
    db_conn.close()

def insert_image_data(image_name, title, buildings, description):
    """Fügt extrahierte Bilddaten in die Datenbank ein."""
    db_conn = sqlite3.connect(DB_FILE)
    cursor = db_conn.cursor()
    try:
        cursor.execute(
            '''
            INSERT INTO image_data (image_name, title, buildings, description)
            VALUES (?, ?, ?, ?)
        ''', (image_name, title, buildings, description)
        )
        db_conn.commit()
        return True # Erfolgreich eingefügt
    except sqlite3.IntegrityError:
        # Dies fängt den Fehler ab, wenn image_name UNIQUE ist und bereits existiert
        st.warning(f"Bild '{image_name}' existiert bereits in der Datenbank und wird nicht erneut hinzugefügt.")
        return False # Nicht eingefügt
    except Exception as e:
        st.error(f"Fehler beim Einfügen von '{image_name}' in die Datenbank: {e}")
        return False
    finally:
        db_conn.close()

def image_exists_in_db(image_name):
    """Überprüft, ob ein Bildname bereits in der Datenbank existiert."""
    db_conn = sqlite3.connect(DB_FILE)
    cursor = db_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM image_data WHERE image_name = ?", (image_name,))
    count = cursor.fetchone()[0]
    db_conn.close()
    return count > 0

# Call database setup once at the start of the app
setup_database()

#%% fields to extract out of image
class MyPictureOutput(BaseModel):
    title: str
    buildings: str
    description: str

# Function to encode the image (angepasst, um Bytes direkt zu verarbeiten)
def encode_image_from_bytes(image_bytes):
    return base64.b64encode(image_bytes).decode('utf-8')

# upload file
uploaded_files = st.file_uploader(label="Bilder hochladen", label_visibility="visible", accept_multiple_files=True)

if uploaded_files:
    # Create a grid for displaying images
    cols = st.columns(4)
    for i, uploaded_file in enumerate(uploaded_files):
        # Zeige das Bild an (unabhängig davon, ob es neu ist oder nicht)
        with cols[i % 4]:
            image_for_display = uploaded_file.read()
            uploaded_file.seek(0) # Setze den Stream-Pointer zurück
            image = Image.open(io.BytesIO(image_for_display))
            st.image(image, caption=uploaded_file.name, use_container_width=True)

        # Überprüfen, ob das Bild bereits in der Datenbank ist
        if image_exists_in_db(uploaded_file.name):
            st.info(f"Bild '{uploaded_file.name}' wurde bereits analysiert und befindet sich in der Datenbank.")
            continue # Springe zum nächsten Bild

        # Process the image and store in DB
        with st.spinner(f"Analysiere '{uploaded_file.name}'..."):
            try:
                # Lese die Bild-Bytes direkt aus dem hochgeladenen Objekt für die LLM-Analyse
                image_bytes_for_llm = uploaded_file.read()
                base64_image = encode_image_from_bytes(image_bytes_for_llm)

                prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            "You are an expert at analyzing images."
                            "Extract the title, buildings, and a description from the image."
                            "Respond with a JSON object with the following keys: title, buildings, description."
                            "For each key (title, buildings, description) you only give a single short line of text. You never use nested JSON objects."
                        ),
                        (
                            "human",
                            [
                                {"type": "text", "text": "Describe the image."},
                                {
                                    "type": "image_url",
                                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                                },
                            ],
                        ),
                    ]
                )

                parser = SimpleJsonOutputParser(pydantic_object=MyPictureOutput)
                chain = prompt | model | parser

                extracted_data = chain.invoke({})

                # Sicherstellen, dass 'buildings' und 'description' Strings sind
                buildings_to_insert = extracted_data.get("buildings", "No buildings identified")
                if isinstance(buildings_to_insert, list):
                    buildings_to_insert = ", ".join(map(str, buildings_to_insert))

                description_to_insert = extracted_data.get("description", "No description")
                if isinstance(description_to_insert, list):
                    description_to_insert = ", ".join(map(str, description_to_insert))

                # Nur einfügen, wenn es noch nicht existiert (wird durch image_exists_in_db und UNIQUE Constraint gehandhabt)
                insert_image_data(
                    image_name=uploaded_file.name,
                    title=extracted_data.get("title", "No title"),
                    buildings=buildings_to_insert,
                    description=description_to_insert,
                )
                # st.success-Meldung wird jetzt von insert_image_data zurückgegeben oder von der outer try-except gefangen.
                

            except Exception as e:
                # Fängt Fehler bei der LLM-Analyse ab
                st.error(f"Fehler bei der Analyse von '{uploaded_file.name}': {e}")

else:
    st.info("Bitte laden Sie JPG-Bilder hoch, um sie hier zu analysieren und anzuzeigen.")

# Download button for the database
if os.path.exists(DB_FILE):
    with open(DB_FILE, "rb") as fp:
        st.download_button(
            label="Download created Database",
            data=fp,
            file_name="buildings.db",
            mime="application/octet-stream"
        )

# %%
