# VisionFlow: Gelişmiş Kişi Takip ve Yeniden Tanıma (Re-ID) Sistemi

VisionFlow, video akışlarında hedeflenen bir kişiyi yüksek doğrulukla takip etmek, hedef kaybolduğunda derin öğrenme (ResNet50) ile kişiyi tekrar tanımak ve hedefe dinamik olarak odaklanmak için geliştirilmiş profesyonel bir bilgisayarlı görü aracıdır.

##  Öne Çıkan Özellikler

* **Tıkla ve Kilitle:** Mouse ile ekrandaki herhangi bir kişiye tıklayarak hedefi anında belirleme.
* **Derin Re-ID Entegrasyonu:** Hedefin görüntüsü kapandığında (occlusion) veya görüşten çıkıp tekrar girdiğinde, **ResNet50** tabanlı özellik çıkarımı ve **Kosinüs Benzerliği** ile hedefi otomatik olarak geri bulma.
* **Dijital Zoom (Oto-Odak):** Hedef kilitlendiği anda görüntü otomatik olarak kişiye odaklanır ve 2x yakınlaştırma yaparak takibi kolaylaştırır.
* **Hibrit Takip:** YOLOv8'in hızı ile derin öğrenmenin tanıma yeteneğini birleştirir.

##  Teknik Altyapı

* **Nesne Tespiti:** YOLOv8 (Ultralytics)
* **Özellik Çıkarımı (Embedding):** ResNet50 (PyTorch)
* **Görüntü İşleme:** OpenCV
* **Arayüz:** Tkinter (Dosya seçimi) & OpenCV HighGUI
* **Benzerlik Hesabı:** Scikit-learn (Cosine Similarity)
