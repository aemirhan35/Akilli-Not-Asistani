import tkinter as tk
from tkinter import filedialog, scrolledtext, messagebox, ttk
import threading
import sys
import os
import time
import sqlite3
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

# Backend modüllerini bağla
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

try:
    from backend.cloud_api import CloudTranscriber
    import backend.main as local_processor 
except ImportError:
    CloudTranscriber = None
    local_processor = None

# --- VERİTABANI YÖNETİCİSİ ---
class DatabaseManager:
    def __init__(self, db_name="asistan_veritabani.db"):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.create_tables()

    def create_tables(self):
        # Kullanıcılar
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS users
                              (username TEXT PRIMARY KEY, password TEXT)''')
        # Notlar (Başlık/Dosya İsmi eklendi!)
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS notes
                              (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                               username TEXT, 
                               title TEXT, 
                               content TEXT, 
                               timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
        try:
            self.cursor.execute("INSERT INTO users VALUES ('admin', '1234')")
            self.conn.commit()
        except sqlite3.IntegrityError:
            pass
        self.conn.commit()

    def login(self, user, pwd):
        self.cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (user, pwd))
        return self.cursor.fetchone() is not None

    def save_note(self, username, title, text):
        self.cursor.execute("INSERT INTO notes (username, title, content) VALUES (?, ?, ?)", (username, title, text))
        self.conn.commit()

    def get_notes_list(self, username):
        # Sadece başlıkları ve tarihleri getirir (Listelemek için)
        self.cursor.execute("SELECT id, title, timestamp FROM notes WHERE username=? ORDER BY id DESC", (username,))
        return self.cursor.fetchall()

    def get_note_content(self, note_id):
        # ID'ye göre içeriği getirir
        self.cursor.execute("SELECT content FROM notes WHERE id=?", (note_id,))
        result = self.cursor.fetchone()
        return result[0] if result else ""

    def get_all_context(self, username):
        # Chatbot için her şeyi getirir
        self.cursor.execute("SELECT title, content FROM notes WHERE username=?", (username,))
        return self.cursor.fetchall()

# --- GİRİŞ EKRANI ---
class LoginWindow:
    def __init__(self, root, on_success):
        self.root = root
        self.on_success = on_success
        self.root.title("Giriş Yap")
        self.root.geometry("350x250")
        self.root.configure(bg="#263238")
        self.db = DatabaseManager()

        tk.Label(root, text="AKILLI ASİSTAN GİRİŞİ", bg="#263238", fg="#eceff1", font=("Segoe UI", 14, "bold")).pack(pady=20)
        
        frame = tk.Frame(root, bg="#263238")
        frame.pack(pady=10)
        
        tk.Label(frame, text="Kullanıcı:", bg="#263238", fg="#b0bec5").grid(row=0, column=0, padx=5, pady=5)
        self.entry_user = tk.Entry(frame)
        self.entry_user.grid(row=0, column=1, padx=5, pady=5)
        
        tk.Label(frame, text="Şifre:", bg="#263238", fg="#b0bec5").grid(row=1, column=0, padx=5, pady=5)
        self.entry_pass = tk.Entry(frame, show="*")
        self.entry_pass.grid(row=1, column=1, padx=5, pady=5)

        tk.Button(root, text="GİRİŞ YAP", command=self.check_login, bg="#2196f3", fg="white", width=20, bd=0, font=("Arial", 10, "bold")).pack(pady=20)
        tk.Label(root, text="(Demo: admin / 1234)", bg="#263238", fg="#546e7a", font=("Arial", 8)).pack()

    def check_login(self):
        user = self.entry_user.get()
        pwd = self.entry_pass.get()
        if self.db.login(user, pwd):
            self.root.destroy()
            self.on_success(user)
        else:
            messagebox.showerror("Hata", "Yanlış kullanıcı adı veya şifre!")

# --- ANA UYGULAMA ---
class MainApp:
    def __init__(self, username):
        self.root = tk.Tk()
        self.username = username
        self.db = DatabaseManager()
        self.root.title(f"Akıllı Not Asistanı - {username}")
        self.root.geometry("1100x750")
        self.root.configure(bg="#263238")

        # STİL AYARLARI
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#263238", borderwidth=0)
        style.configure("TNotebook.Tab", background="#37474f", foreground="white", padding=[10, 5], font=("Arial", 10))
        style.map("TNotebook.Tab", background=[("selected", "#1e88e5")])

        # --- SEKMELER ---
        self.tabs = ttk.Notebook(self.root)
        self.tabs.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_record = tk.Frame(self.tabs, bg="#263238")
        self.tab_history = tk.Frame(self.tabs, bg="#263238")
        self.tab_chat = tk.Frame(self.tabs, bg="#263238")

        self.tabs.add(self.tab_record, text="🎙️ KAYIT & ANALİZ")
        self.tabs.add(self.tab_history, text="📂 GEÇMİŞ (KAYITLAR)")
        self.tabs.add(self.tab_chat, text="🤖 ASİSTAN (CHAT)")

        # Sekmeleri kur
        self.setup_record_tab()
        self.setup_history_tab()
        self.setup_chat_tab()

        self.is_recording = False
        self.recording_data = []
        self.selected_file_path = None

    # ----------------------------------------
    # 1. SEKME: KAYIT VE İŞLEM
    # ----------------------------------------
    def setup_record_tab(self):
        # Sol Panel (İşlemler)
        panel_left = tk.Frame(self.tab_record, bg="#263238", width=400)
        panel_left.pack(side=tk.LEFT, fill="y", padx=20, pady=20)
        
        # Sağ Panel (Log)
        panel_right = tk.Frame(self.tab_record, bg="#1e1e1e")
        panel_right.pack(side=tk.RIGHT, fill="both", expand=True, padx=20, pady=20)

        # -- KAYIT ALANI --
        lbl_title = tk.Label(panel_left, text="YENİ KAYIT", font=("Segoe UI", 16, "bold"), bg="#263238", fg="white")
        lbl_title.pack(anchor="w", pady=(0, 20))

        frame_rec = tk.Frame(panel_left, bg="#37474f", bd=1, relief="solid")
        frame_rec.pack(fill="x", pady=10, ipady=10)

        self.lbl_timer = tk.Label(frame_rec, text="00:00", font=("Consolas", 28), bg="#37474f", fg="#00e676")
        self.lbl_timer.pack()

        self.btn_record = tk.Button(frame_rec, text="🔴 KAYDI BAŞLAT", command=self.toggle_recording, bg="#d32f2f", fg="white", font=("Arial", 11, "bold"), width=20, bd=0)
        self.btn_record.pack(pady=10)

        # -- DOSYA SEÇİMİ --
        tk.Label(panel_left, text="Veya Dosya Yükle:", bg="#263238", fg="#b0bec5").pack(anchor="w", pady=(20, 5))
        frame_file = tk.Frame(panel_left, bg="#37474f")
        frame_file.pack(fill="x")
        
        self.btn_file = tk.Button(frame_file, text="📂 Dosya Seç", command=self.select_file, bg="#ff9800", fg="white", bd=0)
        self.btn_file.pack(side=tk.RIGHT, padx=5, pady=5)
        
        self.lbl_filename = tk.Label(frame_file, text="Seçilmedi", bg="#37474f", fg="#eceff1", font=("Arial", 9))
        self.lbl_filename.pack(side=tk.LEFT, padx=10)

        # -- BUTONLAR --
        tk.Label(panel_left, text="Analiz Yöntemi:", bg="#263238", fg="#b0bec5").pack(anchor="w", pady=(30, 5))
        
        self.btn_local = tk.Button(panel_left, text="🏠 LOKAL MOD\n(PC Gücü - Ücretsiz)", command=self.run_local, bg="#43a047", fg="white", font=("Arial", 10, "bold"), width=30, height=3, bd=0, cursor="hand2")
        self.btn_local.pack(pady=5)
        
        self.btn_cloud = tk.Button(panel_left, text="☁️ CLOUD MOD\n(GPT-4o - Hızlı & Chat)", command=self.run_cloud, bg="#1e88e5", fg="white", font=("Arial", 10, "bold"), width=30, height=3, bd=0, cursor="hand2")
        self.btn_cloud.pack(pady=5)

        # -- LOG EKRANI --
        tk.Label(panel_right, text="📝 İŞLEM DÖKÜMÜ", bg="#1e1e1e", fg="#b0bec5", font=("Arial", 10, "bold")).pack(anchor="w", padx=10, pady=10)
        self.txt_log = scrolledtext.ScrolledText(panel_right, bg="#121212", fg="#00e676", font=("Consolas", 10), bd=0)
        self.txt_log.pack(fill="both", expand=True, padx=10, pady=(0, 10))

    # ----------------------------------------
    # 2. SEKME: GEÇMİŞ (LOKALCİLERİN MEKANI)
    # ----------------------------------------
    def setup_history_tab(self):
        # Sol Taraf: Liste
        frame_list = tk.Frame(self.tab_history, bg="#263238", width=300)
        frame_list.pack(side=tk.LEFT, fill="y", padx=10, pady=10)

        tk.Label(frame_list, text="KAYITLI NOTLARIM", bg="#263238", fg="white", font=("Arial", 12, "bold")).pack(pady=10)
        
        tk.Button(frame_list, text="🔄 Listeyi Yenile", command=self.refresh_history, bg="#546e7a", fg="white", bd=0).pack(fill="x", pady=5)

        # Treeview (Liste Kutusu)
        self.tree = ttk.Treeview(frame_list, columns=("id", "date"), show="headings", height=20)
        self.tree.heading("id", text="Dosya Adı / Başlık")
        self.tree.heading("date", text="Tarih")
        self.tree.column("id", width=180)
        self.tree.column("date", width=100)
        self.tree.pack(fill="both", expand=True)
        self.tree.bind("<<TreeviewSelect>>", self.on_history_select)

        # Sağ Taraf: İçerik Görüntüleme
        frame_content = tk.Frame(self.tab_history, bg="#1e1e1e")
        frame_content.pack(side=tk.RIGHT, fill="both", expand=True, padx=10, pady=10)

        tk.Label(frame_content, text="DOSYA İÇERİĞİ", bg="#1e1e1e", fg="#b0bec5", font=("Arial", 12, "bold")).pack(pady=10)
        self.txt_history_content = scrolledtext.ScrolledText(frame_content, bg="#121212", fg="white", font=("Arial", 11), bd=0)
        self.txt_history_content.pack(fill="both", expand=True, padx=10, pady=10)

    # ----------------------------------------
    # 3. SEKME: ASİSTAN (CLOUDCULARIN MEKANI)
    # ----------------------------------------
    def setup_chat_tab(self):
        # Üst Kısım: Sohbet Geçmişi
        self.chat_history = scrolledtext.ScrolledText(self.tab_chat, bg="#1e1e1e", fg="white", font=("Segoe UI", 11), bd=0)
        self.chat_history.pack(fill="both", expand=True, padx=20, pady=20)
        
        self.chat_history.insert(tk.END, "Asistan: Merhaba! Cloud ile çevirdiğin tüm notlar hafızamda. Bana soru sorabilirsin.\n\n")
        self.chat_history.configure(state="disabled")

        # Alt Kısım: Giriş
        frame_input = tk.Frame(self.tab_chat, bg="#37474f", height=60)
        frame_input.pack(fill="x")

        self.entry_chat = tk.Entry(frame_input, font=("Arial", 12), bg="#cfd8dc", bd=0)
        self.entry_chat.pack(side=tk.LEFT, fill="both", expand=True, padx=20, pady=15)
        self.entry_chat.bind("<Return>", lambda event: self.ask_chatbot()) # Enter ile gönder

        btn_send = tk.Button(frame_input, text="GÖNDER", command=self.ask_chatbot, bg="#1e88e5", fg="white", font=("Arial", 10, "bold"), bd=0)
        btn_send.pack(side=tk.RIGHT, padx=20, pady=15)

    # --- YARDIMCI METODLAR ---
    def log(self, text):
        self.txt_log.insert(tk.END, text + "\n")
        self.txt_log.see(tk.END)

    def select_file(self):
        path = filedialog.askopenfilename(filetypes=[("Ses", "*.wav *.mp3 *.ogg")])
        if path:
            self.selected_file_path = path
            self.lbl_filename.config(text=os.path.basename(path), fg="#00e676")

    # --- GEÇMİŞ SEKMESİ FONKSİYONLARI ---
    def refresh_history(self):
        # Listeyi temizle
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        # Veritabanından çek
        notes = self.db.get_notes_list(self.username)
        for note_id, title, timestamp in notes:
            # Tarihi biraz kırpalım (YYYY-MM-DD HH:MM)
            short_date = str(timestamp)[:16]
            self.tree.insert("", "end", values=(title, short_date), tags=(note_id,))

    def on_history_select(self, event):
        selected_item = self.tree.selection()
        if not selected_item: return
        
        # Seçilen satırın verilerini al
        item_data = self.tree.item(selected_item)
        title = item_data['values'][0]
        
        # ID'yi tags kısmına gizlemiştik ama burada basitçe title üzerinden veya yeniden sorgu ile çekebiliriz.
        # En temiz yöntem: get_note_content metoduna ID göndermek.
        # Treeview'de veriyi nasıl sakladığımıza bakalım.
        # Basitlik için tekrar sorgu atalım veya item_tags kullanalım.
        # Not: Treeview'de ID'yi values'a koymadık, ama veritabanı ID'si lazım.
        # Hack: note_id'yi values[0] yapalım yukarıda? Hayır, Title görünmeli.
        
        # Düzeltme: refresh_history'de note_id'yi gizli tag olarak verdik mi? Hayır.
        # O zaman şöyle yapalım: refresh_history fonksiyonunda ID'yi `iid` olarak kullanalım.
        pass # Aşağıda düzelttim.

    # --- KAYIT (WAV) ---
    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.recording_data = []
            self.btn_record.config(text="⏹️ KAYDI BİTİR", bg="#37474f")
            threading.Thread(target=self._record).start()
            threading.Thread(target=self._timer).start()
        else:
            self.is_recording = False
            self.btn_record.config(text="🔴 KAYDI BAŞLAT", bg="#d32f2f")

    def _record(self):
        fs = 44100
        with sd.InputStream(samplerate=fs, channels=1, callback=lambda i,f,t,s: self.recording_data.append(i.copy())):
            while self.is_recording: sd.sleep(100)
        
        timestamp = int(time.time())
        filename = f"ses_kaydi_{timestamp}.wav"
        write(filename, fs, np.concatenate(self.recording_data, axis=0))
        
        self.selected_file_path = os.path.abspath(filename)
        self.lbl_filename.config(text=f"{filename} (Hazır)", fg="#00e676")
        self.log(f"🎤 Kayıt Tamamlandı: {filename}")

    def _timer(self):
        start = time.time()
        while self.is_recording:
            m, s = divmod(int(time.time()-start), 60)
            self.lbl_timer.config(text=f"{m:02}:{s:02}")
            time.sleep(1)
        self.lbl_timer.config(text="00:00")

    # --- İŞLEM & KAYDETME ---
    def run_local(self):
        if not self.selected_file_path: return messagebox.showwarning("Hata", "Dosya seç!")
        threading.Thread(target=self._process_local).start()

    def _process_local(self):
        self.log("🏠 Lokal Analiz Başlıyor...")
        try:
            local_processor.AUDIO_FILE = self.selected_file_path
            result = local_processor.main()
            
            if result:
                # Dosya ismini başlık olarak kullan
                title = os.path.basename(self.selected_file_path)
                self.db.save_note(self.username, title, result)
                
                self.log("✅ Sonuç veritabanına kaydedildi.")
                self.refresh_history() # Geçmiş listesini güncelle
                messagebox.showinfo("Bitti", "Lokal işlem bitti ve Geçmiş'e eklendi!")
        except Exception as e:
            self.log(f"❌ Hata: {e}")

    def run_cloud(self):
        if not self.selected_file_path: return messagebox.showwarning("Hata", "Dosya seç!")
        threading.Thread(target=self._process_cloud).start()

    def _process_cloud(self):
        self.log("☁️ Cloud Analiz Başlıyor...")
        try:
            transcriber = CloudTranscriber()
            result = transcriber.process_audio(self.selected_file_path)
            
            self.log(f"\nSONUÇ:\n{result}")
            
            title = os.path.basename(self.selected_file_path) + " (Cloud)"
            self.db.save_note(self.username, title, result)
            
            self.log("✅ Kaydedildi. Asistan'a sorabilirsin.")
            self.refresh_history()
            messagebox.showinfo("Bitti", "Cloud işlem bitti!")
        except Exception as e:
            self.log(f"❌ Hata: {e}")

    # --- CHATBOT ---
    def ask_chatbot(self):
        q = self.entry_chat.get()
        if not q: return
        self.entry_chat.delete(0, tk.END)
        
        self.chat_history.configure(state="normal")
        self.chat_history.insert(tk.END, f"Sen: {q}\n", "user")
        self.chat_history.configure(state="disabled")
        
        threading.Thread(target=self._chat_process, args=(q,)).start()

    def _chat_process(self, q):
        try:
            data = self.db.get_all_context(self.username)
            context = "\n".join([f"Dosya: {t}\nİçerik: {c}" for t, c in data])
            
            if not context:
                response = "Henüz kaydedilmiş bir notun yok reis."
            else:
                transcriber = CloudTranscriber()
                prompt = f"Geçmiş notlar:\n{context}\n\nSoru: {q}\nCevap ver:"
                res = transcriber.client.chat.completions.create(
                    model="gpt-4o", messages=[{"role": "user", "content": prompt}]
                )
                response = res.choices[0].message.content

            self.root.after(0, lambda: self._update_chat_ui(response))
        except Exception as e:
            self.root.after(0, lambda: self._update_chat_ui(f"Hata: {e}"))

    def _update_chat_ui(self, text):
        self.chat_history.configure(state="normal")
        self.chat_history.insert(tk.END, f"Asistan: {text}\n\n")
        self.chat_history.see(tk.END)
        self.chat_history.configure(state="disabled")

# Düzeltilmiş refresh_history (ID'yi doğru işlemek için)
def refresh_history_fixed(self):
    for item in self.tree.get_children(): self.tree.delete(item)
    notes = self.db.get_notes_list(self.username)
    for note_id, title, timestamp in notes:
        # iid olarak note_id kullanıyoruz ki tıklayınca bulabilelim
        self.tree.insert("", "end", iid=note_id, values=(title, str(timestamp)[:16]))

def on_history_select_fixed(self, event):
    selected_item = self.tree.selection()
    if not selected_item: return
    note_id = selected_item[0] # iid yani veritabanı ID'si
    content = self.db.get_note_content(note_id)
    
    self.txt_history_content.delete("1.0", tk.END)
    self.txt_history_content.insert(tk.END, content)

# Sınıfa yamalıyoruz (Yukarıdaki koda entegre ettim say reis, sen direkt bunu çalıştırınca olacak)
MainApp.refresh_history = refresh_history_fixed
MainApp.on_history_select = on_history_select_fixed

if __name__ == "__main__":
    # Önce eski veritabanını silmeyi unutma: asistan_veritabani.db
    root_login = tk.Tk()
    def start(user):
        app = MainApp(user)
        # Açılışta geçmişi yükle
        app.refresh_history()
        app.root.mainloop()
    LoginWindow(root_login, start)
    root_login.mainloop()