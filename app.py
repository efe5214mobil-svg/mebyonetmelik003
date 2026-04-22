__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import os
from groq import Groq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="OKUL ARKADAŞIM", page_icon="🏛️", layout="wide")

# --- CSS ---
st.markdown("""
<style>
.main { background-color: #0e1117; }
.stTitle { color: white; text-align: center; font-size: 3rem !important; margin-bottom: 20px; }
.card {
    background-color: #1a1c24;
    border-radius: 15px;
    padding: 20px;
    height: 250px;
    border-top: 5px solid;
    margin-bottom: 20px;
}
.card-red { border-color: #ff4b4b; }
.card-blue { border-color: #0083ff; }
.card-green { border-color: #00d488; }
.card h3 { color: white; margin-bottom: 15px; font-size: 1.2rem; }
.card ul { color: #a3a8b4; list-style-type: none; padding: 0; font-size: 0.9rem; }
.card li { margin-bottom: 8px; }
</style>
""", unsafe_allow_html=True)

# --- BAŞLIK ---
st.markdown("<h1 class='stTitle'>🏛️OKUL ARKADAŞIM</h1>", unsafe_allow_html=True)

# --- API KEY ---
if "GROQ_API_KEY" not in st.secrets:
    st.error("Hata: GROQ_API_KEY bulunamadı!")
    st.stop()

# --- KARTLAR ---
st.markdown("### 💡 Hızlı Sorular")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""<div class="card card-red"><h3>📜 Kayıt & Disiplin</h3><ul>
    <li>• Disiplin cezaları nelerdir?</li>
    <li>• Kayıt işlemleri nasıl yapılır?</li>
    <li>• Kınama cezası dosyaya işlenir mi?</li></ul></div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""<div class="card card-blue"><h3>⌛ Devamsızlık</h3><ul>
    <li>• 10/30 gün kuralı nedir?</li>
    <li>• Ortalamam 75 ve 8 gün devamsızlıkla belge alabilir miyim?</li>
    <li>• 11 gün özürsüz devamsızlıkta kalır mıyım?</li></ul></div>""", unsafe_allow_html=True)

with col3:
    st.markdown("""<div class="card card-green"><h3>🎓 Başarı</h3><ul>
    <li>• Teşekkür belgesi kaç puan?</li>
    <li>• 48 ortalama ile geçilir mi?</li>
    <li>• 86 ortalama takdir alır mı?</li></ul></div>""", unsafe_allow_html=True)

st.markdown("---")

# --- VECTOR DB ---
@st.cache_resource
def load_existing_vector_db():
    persist_dir = "okul_asistani_v2_db"
    if not os.path.exists(persist_dir):
        return None
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

v_db = load_existing_vector_db()

# --- AI CEVAP ---
def ask_asistant(v_db, query):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    if v_db:
        docs = v_db.similarity_search(query, k=5)
        baglam = "\n\n".join([doc.page_content for doc in docs])
    else:
        baglam = "Veri bulunamadı"

    system_msg = """Sen MEB Mevzuat Asistanısın. Yanıtların ÇOK KISA (en fazla 2 cümle) ve NET olmalı.
ASLA DEĞİŞMEZ ANALİZ KURALLARI:
1. SORUMLULUK: Sorumluluk sınavı geçme puanı 50'dir.
2. MANTIK: 8 sayısı 10'dan küçüktür; 8 gün devamsızlıkla kalınmaz.
3. GÜNCEL: Devamsızlık artık başarı belgesi almaya engel DEĞİLDİR.
4. RAPOR: Hastane raporları 'Özürlü' devamsızlıktır.
5. SINIF GEÇME: 3 dersten fazla zayıfı olan KALIR.
6. MATEMATİK: Ortalama 50+ ise "Evet geçebilirsin" diye başla.
7. BELGE: Teşekkür 70-84.99, Takdir 85+
8. 50 ve üzeri not alan GEÇER.
9. Çelişki varsa 50 kuralını uygula.
TALİMAT: Sadece cevabı ver."""

    chat = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"Bağlam: {baglam}\n\nSoru: {query}"}
        ],
        model="llama-3.1-8b-instant",
        temperature=0
    )

    return chat.choices[0].message.content

# --- CHAT ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# INPUT HER ZAMAN VAR
if prompt := st.chat_input("Yönetmelik hakkında bir soru sorun..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = ask_asistant(v_db, prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# --- UYARI ---
if not v_db:
    st.warning("⚠️ Vector veritabanı bulunamadı. Sistem sınırlı çalışıyor.")st.markdown("<h1 class='stTitle'>🏛️ OKUL ARKADAŞIM</h1>", unsafe_allow_html=True)

# --- API KEY KONTROL ---
if "GROQ_API_KEY" not in st.secrets:
    st.error("GROQ_API_KEY bulunamadı!")
    st.stop()

# --- KARTLAR ---
st.markdown("### 💡 Hızlı Sorular")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""<div class="card card-red"><h3>📜 Kayıt & Disiplin</h3><ul>
    <li>• Disiplin cezaları nelerdir?</li>
    <li>• Kayıt işlemleri nasıl yapılır?</li>
    <li>• Kınama cezası dosyaya işlenir mi?</li></ul></div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""<div class="card card-blue"><h3>⌛ Devamsızlık</h3><ul>
    <li>• 10/30 gün kuralı nedir?</li>
    <li>• 8 gün devamsızlıkla belge alınır mı?</li>
    <li>• 11 gün özürsüz devamsızlıkta kalınır mı?</li></ul></div>""", unsafe_allow_html=True)

with col3:
    st.markdown("""<div class="card card-green"><h3>🎓 Başarı</h3><ul>
    <li>• Teşekkür kaç puan?</li>
    <li>• 48 ortalama ile geçilir mi?</li>
    <li>• 86 ortalama takdir alır mı?</li></ul></div>""", unsafe_allow_html=True)

st.markdown("---")

# --- VECTOR DB YÜKLEME ---
@st.cache_resource
def load_db():
    path = "okul_asistani_v2_db"
    if not os.path.exists(path):
        return None
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=path, embedding_function=embeddings)

v_db = load_db()

# --- CEVAP FONKSİYONU ---
def ask(v_db, query):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])

    if v_db:
        docs = v_db.similarity_search(query, k=5)
        baglam = "\n\n".join([d.page_content for d in docs])
    else:
        baglam = "Veri yok"

    system_msg = """ÇOK kısa cevap ver (max 2 cümle).
Kurallar:
- 50 ve üzeri geçer
- 3'ten fazla zayıf kalır
- Teşekkür 70-84, Takdir 85+
- Devamsızlık belgeye engel değil"""

    chat = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": f"{baglam}\nSoru: {query}"}
        ],
        model="llama-3.1-8b-instant",
        temperature=0
    )

    return chat.choices[0].message.content

# --- CHAT SİSTEMİ ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Eski mesajlar
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# INPUT (HER ZAMAN GÖRÜNÜR)
if prompt := st.chat_input("Sorunu yaz..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = ask(v_db, prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})

# --- ALT UYARI ---
if not v_db:
    st.warning("⚠️ Vector veritabanı yok, sistem sınırlı çalışıyor.")st.markdown("<h1 class='stTitle'>🏛️OKUL ARKADAŞIM</h1>", unsafe_allow_html=True)

# --- SECRETS KONTROLÜ ---
if "GROQ_API_KEY" not in st.secrets:
    st.error("Hata: GROQ_API_KEY Streamlit Secrets panelinde bulunamadı!")
    st.stop()

# --- HIZLI SORULAR (GÖRSEL KARTLAR) ---
st.markdown("### 💡 Hızlı Sorular")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""<div class="card card-red"><h3>📜 Kayıt & Disiplin</h3><ul>
    <li>• Disiplin cezaları nelerdir?</li><li>•Kayıt işlemleri nasıl yapılır?</li>
    <li>• "Kınama" cezası alan öğrencinin dosyasına işlenir mi?</li></ul></div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""<div class="card card-blue"><h3>⌛ Devamsızlık</h3><ul>
    <li>• 10/30 gün kuralı nedir?</li><li>• Ortalamam 75 ve toplam 8 gün devamsızlığım var, belge alabilir miyim?</li>
    <li>• 11 gün özürsüz devamsızlığım var, sınıfta kalır mıyım?</li></ul></div>""", unsafe_allow_html=True)

with col3:
    st.markdown("""<div class="card card-green"><h3>🎓 Başarı </h3><ul>
    <li>• Teşekkür belgesi alabilmek için ortalamamın kaç olması lazım?</li><li>• Yıl sonu başarı ortalamam 48, sınıfı geçebilir miyim?</li>
    <li>• 86 puan ortalamam var, devamsızlığım olsa da Takdir alabilir miyim?</li></ul></div>""", unsafe_allow_html=True)

st.markdown("---")

# --- 1. KAYITLI VEKTÖR DOSYASINI YÜKLEME ---
@st.cache_resource
def load_existing_vector_db():
    persist_dir = "okul_asistani_v2_db" 
    if not os.path.exists(persist_dir):
        st.error(f"Hata: '{persist_dir}' klasörü GitHub deponuzda bulunamadı!")
        return None
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# --- 2. CEVAP ÜRETME ---
def ask_asistant(v_db, query):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    docs = v_db.similarity_search(query, k=5)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    # Colab'daki başarılı kural seti
    system_msg = """Sen MEB Mevzuat Asistanısın. Yanıtların ÇOK KISA (en fazla 2 cümle) ve NET olmalı.
    ASLA DEĞİŞMEZ ANALİZ KURALLARI:
    1. SORUMLULUK: Sorumluluk sınavı geçme puanı 50'dir.
    2. MANTIK: 8 sayısı 10'dan küçüktür; 8 gün devamsızlıkla kalınmaz.
    3. GÜNCEL: Devamsızlık artık başarı belgesi (Takdir/Teşekkür) almaya engel DEĞİLDİR.
    4. RAPOR: Hastane raporları 'Özürlü' devamsızlıktır.
    5. SINIF GEÇME: Ortalaması 50 olsa bile 3 dersten fazla zayıfı olan öğrenci KALIR.
    6. MATEMATİKSEL ONAY: Eğer ortalama 50 ve üzerindeyse (Örn: 52), söze "Evet geçebilirsin" diyerek başla ve "Ancak zayıf sayın 3'ten az olmalıdır" şartını hatırlat.
    7. BELGE: Teşekkür 70-84.99, Takdir 85.00 ve üzeri ortalama gerektirir.
    8. Eğer öğrenci 50 ve üzerinde bir not almışsa KESİNLİKE GEÇMİŞTİR.
    9. Bağlamda çelişkili rakamlar görürsen, her zaman '50 ve üzeri geçer' kuralını uygula.
    TALİMAT: Sadece sorunun cevabını ver. Gereksiz açıklama yapma."""

    

    chat = client.chat.completions.create(
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": f"Bağlam: {baglam}\n\nSoru: {query}"}],
        model="llama-3.1-8b-instant",
        temperature=0
    )
    return chat.choices[0].message.content

# --- SOHBET AKIŞI ---
v_db = load_existing_vector_db()

if v_db:
    # Session state kontrolü ve mesajların saklanması
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Eski mesajları ekrana bas
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Yeni soru girişi
    if prompt := st.chat_input("Yönetmelik hakkında bir soru sorun..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            response = ask_asistant(v_db, prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

yazı yazma yeri çıkmıyor hızlı fixst.markdown("<h1 class='stTitle'>🏛️OKUL ARKADAŞIM</h1>", unsafe_allow_html=True)

# --- SECRETS KONTROLÜ ---
if "GROQ_API_KEY" not in st.secrets:
    st.error("Hata: GROQ_API_KEY Streamlit Secrets panelinde bulunamadı!")
    st.stop()

# --- HIZLI SORULAR (GÖRSEL KARTLAR) ---
st.markdown("### 💡 Hızlı Sorular")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""<div class="card card-red"><h3>📜 Kayıt & Disiplin</h3><ul>
    <li>• Disiplin cezaları nelerdir?</li><li>•Kayıt işlemleri nasıl yapılır?</li>
    <li>• "Kınama" cezası alan öğrencinin dosyasına işlenir mi?</li></ul></div>""", unsafe_allow_html=True)

with col2:
    st.markdown("""<div class="card card-blue"><h3>⌛ Devamsızlık</h3><ul>
    <li>• 10/30 gün kuralı nedir?</li><li>• Ortalamam 75 ve toplam 8 gün devamsızlığım var, belge alabilir miyim?</li>
    <li>• 11 gün özürsüz devamsızlığım var, sınıfta kalır mıyım?</li></ul></div>""", unsafe_allow_html=True)

with col3:
    st.markdown("""<div class="card card-green"><h3>🎓 Başarı </h3><ul>
    <li>• Teşekkür belgesi alabilmek için ortalamamın kaç olması lazım?</li><li>• Yıl sonu başarı ortalamam 48, sınıfı geçebilir miyim?</li>
    <li>• 86 puan ortalamam var, devamsızlığım olsa da Takdir alabilir miyim?</li></ul></div>""", unsafe_allow_html=True)

st.markdown("---")

# --- 1. KAYITLI VEKTÖR DOSYASINI YÜKLEME ---
@st.cache_resource
def load_existing_vector_db():
    persist_dir = "okul_asistani_v2_db" 
    if not os.path.exists(persist_dir):
        st.error(f"Hata: '{persist_dir}' klasörü GitHub deponuzda bulunamadı!")
        return None
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# --- 2. CEVAP ÜRETME ---
def ask_asistant(v_db, query):
    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    docs = v_db.similarity_search(query, k=5)
    baglam = "\n\n".join([doc.page_content for doc in docs])

    # Colab'daki başarılı kural seti
    system_msg = """Sen MEB Mevzuat Asistanısın. Yanıtların ÇOK KISA (en fazla 2 cümle) ve NET olmalı.
    ASLA DEĞİŞMEZ ANALİZ KURALLARI:
    1. SORUMLULUK: Sorumluluk sınavı geçme puanı 50'dir.
    2. MANTIK: 8 sayısı 10'dan küçüktür; 8 gün devamsızlıkla kalınmaz.
    3. GÜNCEL: Devamsızlık artık başarı belgesi (Takdir/Teşekkür) almaya engel DEĞİLDİR.
    4. RAPOR: Hastane raporları 'Özürlü' devamsızlıktır.
    5. SINIF GEÇME: Ortalaması 50 olsa bile 3 dersten fazla zayıfı olan öğrenci KALIR.
    6. MATEMATİKSEL ONAY: Eğer ortalama 50 ve üzerindeyse (Örn: 52), söze "Evet geçebilirsin" diyerek başla ve "Ancak zayıf sayın 3'ten az olmalıdır" şartını hatırlat.
    7. BELGE: Teşekkür 70-84.99, Takdir 85.00 ve üzeri ortalama gerektirir.
    8. Eğer öğrenci 50 ve üzerinde bir not almışsa KESİNLİKE GEÇMİŞTİR.
    9. Bağlamda çelişkili rakamlar görürsen, her zaman '50 ve üzeri geçer' kuralını uygula.
    TALİMAT: Sadece sorunun cevabını ver. Gereksiz açıklama yapma."""

    

    chat = client.chat.completions.create(
        messages=[{"role": "system", "content": system_msg},
                  {"role": "user", "content": f"Bağlam: {baglam}\n\nSoru: {query}"}],
        model="llama-3.1-8b-instant",
        temperature=0
    )
    return chat.choices[0].message.content

# --- SOHBET AKIŞI ---
v_db = load_existing_vector_db()

# Mesaj geçmişini başlat (v_db'den bağımsız)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Eski mesajları ekrana bas
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Giriş alanı (Her zaman görünür)
prompt = st.chat_input("Yönetmelik hakkında bir soru sorun...")

if prompt:
    # Kullanıcı mesajını ekle ve göster
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Cevap üretme kısmı
    with st.chat_message("assistant"):
        if v_db is not None:
            response = ask_asistant(v_db, prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            error_msg = "Veritabanı yüklenemediği için şu an cevap veremiyorum. Lütfen klasör yolunu kontrol edin."
            st.error(error_msg)
