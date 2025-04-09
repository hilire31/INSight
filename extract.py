import nltk
from PyPDF2 import PdfReader
#nltk.download('punkt_tab')
# Télécharger le tokenizer de phrases si nécessaire
#nltk.download('punkt')

from nltk.tokenize import sent_tokenize

def pdf_to_sentences(pdf_path, txt_output_path):
    # Lire le PDF
    reader = PdfReader(pdf_path)
    full_text = ""

    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    
    
    
    paragraphs = full_text.split("\n")

    paragraphs=[i for i in paragraphs if len(i)>15]
    paragraphs = [
    "".join(paragraphs[i:i+3])
    for i in range(0, len(paragraphs), 3)
    ]
    
    """full_text=""
    for p in paragraphs:
        if len(p) > 20:  # pour éviter les lignes vides ou très courtes
            full_text+= p + "\n\n"


    # Découper le texte en phrases
    sentences = sent_tokenize(full_text,language="french")
    grouped_sentences = [
    " ".join(sentences[i:i+9])
    for i in range(0, len(sentences), 9)
    ]
    """
    small_to_big=True

    # Écrire chaque phrase dans un fichier texte
    with open(txt_output_path, "w", encoding="utf-8") as f:
        if small_to_big:
            for i in range(len(paragraphs)):
                big=""
                if len(paragraphs[i].strip())>20:
                    if i==0 or i==len(paragraphs)-1:
                        big=paragraphs[i].strip()
                    else:
                        big=paragraphs[i-1].strip()+paragraphs[i].strip()+paragraphs[i+1].strip()
                    f.write(paragraphs[i].strip() + "\t"+big+"\n")
        else:
            f.write(paragraphs[i].strip() + "\t"+paragraphs[i].strip()+"\n")



# Exemple d'utilisation
pdf_to_sentences("Reglement_des_Etudes_2023-2024.pdf", "reglement.txt")
