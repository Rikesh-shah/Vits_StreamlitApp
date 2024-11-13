import streamlit as st
import torch
import numpy as np
import io
import soundfile as sf
from transformers import VitsModel, AutoTokenizer
from tts import vist_voice

# Set page configuration for title and icon
st.set_page_config(page_title="VITS Text-to-Speech App", page_icon="🔊")


phrases = [
"Goedemorgen. Ik ben Annemarie, de digitale medewerker. Om u goed en snel van dienst te zijn, vragen wij u om een aantal vragen te beantwoorden. Hierdoor kunnen mijn collega's u beter helpen. Wilt u mijn vragen beantwoorden?",
"Goedemiddag. Ik ben Annemarie, de digitale medewerker. Om u goed en snel van dienst te zijn, vragen wij u om een aantal vragen te beantwoorden. Hierdoor kunnen mijn collega's u beter helpen. Wilt u mijn vragen beantwoorden?",
"Goedenavond. Ik ben Annemarie, de digitale medewerker. Om u goed en snel van dienst te zijn, vragen wij u om een aantal vragen te beantwoorden. Hierdoor kunnen mijn collega's u beter helpen. Wilt u mijn vragen beantwoorden?",
"Dat is jammer, wij kunnen u dan sneller helpen. Wilt u toch meewerken?",
"Sorry, ik heb u niet begrepen. Ik verwacht een ja of nee antwoord.",
"Fijn. Mag ik uw naam noteren?",
"Ik heb genoteerd Bas schiltmas . Is dit correct?",
"Ik heb uw naam niet goed verstaan. Kunt u deze herhalen?",
"Op deze vraag verwacht ik uw naam als antwoord. Kunt u mij uw naam geven?",
"Dat is jammer, wij kunnen u dan sneller helpen. Wilt u toch meewerken?",
"Mag ik uw naam noteren?",
"Op deze vraag verwacht ik een ja of nee antwoord. Kunt u mij dit geven?",
"Mag ik dan uw telefoonnummer noteren?",
"Ik heb genoteerd twee twee drie vier vijf zes zeven acht negen nul . Is dit correct?",
"Kunt u mij uw telefoonnummer geven in plaats van ja of nee?",
"Dat is jammer. Als u meewerkt, kunnen wij u sneller helpen. Wilt u toch meewerken?",
"Het nummer dat u gaf is niet correct. Kunt u mij het juiste telefoonnummer geven?",
"Ik heb uw antwoord niet begrepen. Kunt u het nogmaals proberen?",
"Mag ik uw telefoonnummer noteren?",
"Mag ik dan nu uw postcode noteren?",
"Ik heb uw postcode genoteerd. Is vier vijf één één X C de juiste postcode?",
"Kunt u mij uw postcode geven in plaats van ja of nee?",
"Dat is jammer, dit helpt ons om u sneller te helpen. Wilt u toch meewerken?",
"De postcode die u gaf is geen geldige postcode. Kunt u deze herhalen?",
"Ik heb uw antwoord niet begrepen. Kunt u het nogmaals proberen?",
"Mag ik uw postcode noteren?",
"Mag ik uw huisnummer noteren?",
"Ik heb uw huisnummer genoteerd. Is één het juiste huisnummer?",
"Kunt u mij uw huisnummer geven in plaats van ja of nee?",
"Dat is jammer, dit helpt ons om u sneller te helpen. Wilt u toch meewerken?",
"Het huisnummer dat u gaf lijkt niet te bestaan. Kunt u uw huisnummer herhalen?",
"Mag ik uw huisnummer noteren?",
" Ik kan dit huisnummer niet vinden in combinatie met deze postcode",
"Fijn dat ik uw gegevens heb mogen noteren, kunt u mij vertellen waarvoor u belt?",
"Ik heb voor u genoteerd: gas lekkage is dit correct?",
"Dat is jammer, dit helpt ons om u sneller te helpen. Wilt u toch meewerken?",
"Ik begrijp niet precies wat u bedoelt. Kunt u mij nogmaals de reden geven waarvoor u belt?",
"Kunt u ons vertellen waarvoor u belt?",
"Dank u voor deze informatie. Wij geven dit zo spoedig mogelijk door aan de eerstvolgende beschikbare medewerker. Deze zal u verder helpen.",
"Helaas heb ik u niet begrepen. Ik zal u doorverbinden met de eerst beschikbare medewerker.",
"Hartelijk bedankt voor uw tijd.",
"Sorry, ik heb het niet verstaan. Kunt u dat alstublieft herhalen?",
]

# Set up models and tokenizers (ensure correct paths are available)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

dutch_model = VitsModel.from_pretrained("models/voice_dutch/female_nl_1.0.6", local_files_only=True)
dutch_tokenizer = AutoTokenizer.from_pretrained("models/voice_dutch/female_nl_1.0.6", local_files_only=True)
dutch_model.to(device)

# english_model = VitsModel.from_pretrained("models/voice_english/female_en", local_files_only=True)
# english_tokenizer = AutoTokenizer.from_pretrained("models/voice_english/female_en", local_files_only=True)
# english_model.to(device)

# Streamlit app layout
st.title("Text-to-Speech with VITS Model")

with st.expander("Press here to view the text list :"):
    st.write("Here is the list of System Text Lists:")
    for i, phrase in enumerate(phrases,start=1):
        st.write(f"{i}.{phrase}")

# User selection: pre-defined list or custom input
choice = st.radio("Choose input type:", ("Use Predefined List", "Enter Custom Text"))

# Get selected language
language = st.selectbox("Select Language:", ("Dutch","English"))

if choice == "Use Predefined List":
    # Generate audio for each phrase in the list
    if st.button("Generate Audio for All Phrases"):
        with st.spinner("Generating audio..."):
            for i, text in enumerate(phrases):
                if language == "Dutch":
                    model = dutch_model
                    tokenizer = dutch_tokenizer
                else:
                    # model = english_model
                    # tokenizer = english_tokenizer
                    pass

                # Run the text-to-speech function
                audio_data = vist_voice(text, tokenizer, model)

                # Save audio to a buffer and use Streamlit's audio player to play it
                buffer = io.BytesIO()
                sf.write(buffer, audio_data, samplerate=16000, format='WAV')
                buffer.seek(0)

                # Display audio with phrase number
                st.write(f"Phrase {i+1}: {text}")
                st.audio(buffer, format="audio/wav")

else:
    # Custom text input
    user_text = st.text_input("Enter text to synthesize:")
    
    if st.button("Generate Audio"):
        if user_text:
            with st.spinner("Generating audio..."):
                if language == "Dutch":
                    model = dutch_model
                    tokenizer = dutch_tokenizer
                else:
                    # model = english_model
                    # tokenizer = english_tokenizer
                    pass

                # Run the text-to-speech function
                audio_data = vist_voice(user_text, tokenizer, model)

                # Save audio to a buffer and use Streamlit's audio player to play it
                buffer = io.BytesIO()
                sf.write(buffer, audio_data, samplerate=16000, format='WAV')
                buffer.seek(0)

                # Play the audio
                st.audio(buffer, format="audio/wav")
        else:
            st.warning("Please enter some text.")
