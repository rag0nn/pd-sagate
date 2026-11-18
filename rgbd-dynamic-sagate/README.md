# Fizik-Öncüllü ve QMF-Kalibreli Dinamik SA-Gate (`PD-SAGate`) ile Dayanıklı RGB-D Anlamsal Bölütleme

Bu proje, **ESANet-R34-NBt1D** omurgası üzerinde sensör bozulmalarına (eksik derinlik delikleri $D=0$, Kinect aksiyal $z^2$ gürültüsü, düşük ışık, hareket bulanıklığı ve çapraz modalite kayıpları) karşı dayanıklı ve hesaplama açısından verimli bir RGB-D füzyon mimarisi (**`PD-SAGate`**) sunar.

## Temel Bileşenler ve Kuramsal Dayanaklar
1. **Fiziksel Derinlik Kamerası Önseli (`DepthPrior`):** Ham derinlik haritasından türevlenemez (`@torch.no_grad()`) olarak çıkarılan ölü piksel oranı $M_{\text{hole}} = \mathbb{I}(D \le 10^{-3})$ ve geçerli komşuluklardaki Sobel süreksizlik gradyanı $\|\nabla D\|_2$, öğrenilebilir pozitif katsayılarla ($\alpha, \beta > 0$) birleştirilerek fiziksel derinlik cezası $P_{\text{depth}}(u,v) = \alpha M_{\text{hole}}(u,v) + \beta \|\nabla D(u,v)\|_2$ oluşturulur.
2. **Fizik-Destekli Kanal Ayırma ve Piksel-Seviyesi Softmax Toplama (*SA-Gate*, Chen vd. ECCV 2020):**
   - **Adım 1 (Ayırma):** Küresel öznitelik özetleri ve fiziksel sensör istatistikleri (`Mean(M_hole)`, `Mean(||∇D||_2)`) ile her iki modalitenin kanalları yeniden kalibre edilir ($\tilde{r}, \tilde{d}$).
   - **Adım 2 (Softmax Toplama):** Piksel bazında kalite logitlerinden fiziksel ceza düşülerek ($z_{\text{depth}}^{\text{eff}} = z_{\text{depth}} - P_{\text{depth}}$) iki modalite arasında **bölüşüm birliği** ($w_{\text{rgb}} + w_{\text{depth}} = 1$) sağlayan Softmax ağırlıkları hesaplanır.
3. **QMF Teorem-1 Kalite-Tutarlılığı ve Negatif Kovaryans Kaybı (*QMF*, Zhang vd. ICML 2023):**
   - Eğitim sırasında son kodlayıcı aşamasındaki ($s=4$) ayrılmış RGB ve Depth özniteliklerine uygulanan hafif `1×1 Conv` yardımcı başlıklar (`aux_head_rgb`, `aux_head_depth`) ile örnek başına tekil modalite kayıpları ($\ell_{\text{rgb}}^{(b)}, \ell_{\text{depth}}^{(b)}$) ölçülür ve **negatif kovaryans sıralama kaybı** ($\text{Cov}(w_m, \ell_m) < 0$) ile daha yüksek hata üreten modalitenin ağırlığı otomatik olarak bastırılır.
4. **DynMM Koşullu Çapraz İyileştirme Atlaması (*DynMM*, Xue & Marculescu CVPRW 2023):**
   - Görüntü bazında ortalama derinlik güveni $\bar{w}_{\text{depth}} < \tau$ ($\tau = 0.28$) olduğunda ağır çapraz iyileştirme (`Refine`) bloğu çıkarım sırasında gerçekten atlanarak GFLOPs ve gecikme tasarrufu sağlanır.

## Dizin Yapısı
- `src/models/pd_sagate_fusion.py`: `DepthPrior`, `SpatialQualityHead` ve `PDSAGateFusion` (`mask_only`, `phys_sagate`, `full` modları).
- `src/models/esanet_pd_sagate.py`: `ESANetPDSAGate` omurga sarıcısı ve eğitim-içi QMF yardımcı başlıkları.
- `src/models/losses.py`: `MultiScaleWeightedCE`, `QMFQualityLoss` ve `TotalPDSAGateLoss`.
- `src/datasets/`: `nyuv2_dataset.py` (795 train / 654 test) ve `corruptions.py` (11 değerlendirme senaryosu).
- `scripts/`:
  - `eval_baseline.py`: V1 (`ESANet-SE`) baseline doğrulaması.
  - `train_pdsagate.py`: Dengeli Paket (`--balanced_pack`) veya tekil varyant eğitimi.
  - `run_ablations.py`: V1..V4 11-senaryo ablasyon, QMF ampirik kovaryans ve maliyet ölçümü.
  - `generate_plots_and_tables.py`: 11-senaryo bozulma ızgarası, mimari şema, dayanıklılık eğrileri ve LaTeX tabloları.
- `tests/test_pdsagate_and_corruptions.py`: Birim testleri.
