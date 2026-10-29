# Problem Setting: Causal Tracing theo trục denoising để định vị Safety-Critical States trong Diffusion LLMs

> Ghi chú venue: [A*] = đã xác nhận tại venue A*; [arXiv] = preprint; [A*, theo arXiv] = venue chỉ lấy từ comment/journal-ref trên arXiv, cần đối chiếu lại. Các con số là số do tác giả báo cáo.

---

## 1. Bối cảnh

Diffusion LLMs (dLLMs) dạng masked như LLaDA, LLaDA-1.5, Dream và MMaDA sinh văn bản theo cách khác hẳn LLM tự hồi quy (AR). Model bắt đầu từ một chuỗi phản hồi toàn [MASK], rồi qua $K$ bước denoising nó lần lượt *commit* các token ở những vị trí bất kỳ, nhờ bidirectional attention và parallel decoding. Vì vậy hành vi an toàn (refusal) của dLLM không chỉ phụ thuộc vào *vị trí* token mà còn vào *thời điểm* (bước denoising) mà một trạng thái nội tại hình thành rồi được commit thành output. Các công trình dưới đây đã lần lượt định vị safety trong AR LLM, trong diffusion ảnh, và (về mặt hành vi) trong dLLM. Tuy vậy, vẫn chưa có công trình nào trả lời trực tiếp câu hỏi: *trạng thái nội tại nào, ở layer nào, bước denoising nào và vị trí nào, là nguyên nhân quyết định model từ chối hay tuân thủ một yêu cầu độc hại?*

---

## 2. Các phương pháp liên quan

### 2.1. Refusal Direction: Arditi et al., *Refusal in Language Models Is Mediated by a Single Direction* (NeurIPS 2024) [A*]

**Đã giải quyết được gì.** Arditi et al. cho thấy hành vi từ chối của 13 chat model mã nguồn mở (đến 72B tham số) được trung gian bởi một hướng duy nhất trong residual stream. Hướng này được tính bằng difference-in-means giữa activation của prompt độc hại và prompt vô hại, tại các vị trí token sau instruction. Bằng chứng mang tính can thiệp (interventional) ở cả hai chiều. Chiếu bỏ hướng này khỏi mọi layer và vị trí (directional ablation) làm mất khả năng từ chối, tức là bằng chứng về tính cần (necessity). Cộng hướng này vào lại khiến model từ chối cả prompt vô hại, tức là bằng chứng về tính đủ (sufficiency). Từ đó tác giả xây dựng một white-box jailbreak bằng weight orthogonalization, và chỉ ra rằng adversarial suffix hoạt động bằng cách chặn sự lan truyền của hướng refusal. Đây là kết quả nền tảng, đặt safety vào một đối tượng hình học cụ thể, có thể đo và can thiệp được.

**Chưa giải quyết được gì.** Phân tích được thiết kế cho một forward pass của AR model, nên không có trục thời gian, và hành vi được đánh giá chủ yếu qua refusal score ở các token đầu của output. Hướng được chọn bằng heuristic, và công trình không giải thích hướng đó được tính ra từ đâu ở upstream. Các nghiên cứu sau còn làm lung lay giả định "một hướng duy nhất". Wollschläger et al. (2025, venue cần xác minh) tìm thấy nhiều hướng refusal tạo thành concept cone, và chỉ ra rằng trực giao không đồng nghĩa với độc lập khi can thiệp. Zhao et al. (2025) lập luận rằng harmfulness và refusal được mã hóa tách biệt. Quan trọng nhất với dLLM, một bài workshop ICLR 2026 báo cáo rằng refusal direction không chuyển được từ AR sang masked diffusion model. Như vậy, câu hỏi liệu dLLM có một hướng (hay cone) refusal hay không, và hình học của nó biến đổi thế nào qua các bước denoising, vẫn còn bỏ ngỏ.

### 2.2. Shallow Safety Alignment: Qi et al., *Safety Alignment Should Be Made More Than Just a Few Tokens Deep* (ICLR 2025, Outstanding Paper) [A*]

**Đã giải quyết được gì.** Qi et al. đo per-token KL divergence giữa model đã align và model base. Kết quả cho thấy alignment chủ yếu thay đổi phân phối của vài token đầu tiên của phản hồi, hiện tượng được gọi là "shallow safety alignment". Phát hiện này giải thích thống nhất vì sao prefilling attack, adversarial suffix, tấn công qua tham số decoding và fine-tuning attack đều hiệu quả: chỉ cần vượt qua vài token đầu là model tiếp tục sinh nội dung độc hại. Tác giả cũng đề xuất hai cách khắc phục: data augmentation với các mẫu "safety recovery" (một prefix độc hại rồi quay về từ chối), và một hàm mục tiêu fine-tuning có ràng buộc theo từng token nhằm hạn chế sự trôi ở các token đầu.

**Chưa giải quyết được gì.** Bằng chứng nằm ở mức phân phối output (hành vi), không định vị trạng thái nội tại nào mang "độ sâu" an toàn. Khái niệm "vài token đầu" mặc định thứ tự sinh trái sang phải, nên không còn được định nghĩa rõ trong dLLM, nơi token có thể được commit theo bất kỳ thứ tự nào. Các công trình dLLM gần đây gợi ý rằng phiên bản tương ứng là "shallow-step", tức nông theo *bước denoising* chứ không chỉ theo *vị trí*. Tuy nhiên, chưa có phân tích nhân quả nào chỉ ra cơ chế nội tại của độ nông theo bước này. Các cách khắc phục của Qi et al. cũng mới chỉ là giải pháp một phần.

### 2.3. LOCA: Kumar & Ahuja, *Minimal, Local, Causal Explanations for Jailbreak Success in LLMs* (2026) [arXiv]

**Đã giải quyết được gì.** LOCA đại diện cho SOTA hiện nay về causal localization ở mức tinh cho jailbreak trên AR LLM. Phương pháp ghép token giữa prompt độc hại và phiên bản jailbreak của nó, sau đó chỉ patch activation dọc theo các decoder direction của sparse autoencoder (SAE). Mỗi cặp (token, direction) được chấm điểm bằng một xấp xỉ bậc nhất và được tính lại lặp đi lặp lại. Với chỉ khoảng 6–8 patch trên Llama-3.1-8B (12–16 trên Gemma-2-2B), LOCA khôi phục được hành vi từ chối, trong khi các baseline thường thất bại kể cả sau 20 patch. Kết quả cũng mang tính diễn giải: patch ở layer giữa rơi vào các token template sau instruction và dấu câu. Điều này cho thấy jailbreak hoạt động bằng cách khuếch đại các khái niệm "vô hại" tại những vị trí đó.

**Chưa giải quyết được gì.** Mục tiêu tối ưu của LOCA là KL divergence trên *token output đầu tiên*. Proxy này thất bại khi phản hồi gốc và phản hồi bị jailbreak có cùng token đầu, và nó bỏ sót nội dung độc hại xuất hiện muộn hơn. Phương pháp chỉ hoạt động trên lưới (layer × vị trí prompt) của một forward pass AR, nên không có khái niệm bước sinh hay bước denoising. Ngoài ra nó phụ thuộc vào chất lượng SAE, chạy chậm, và cũng chịu các rủi ro phương pháp luận chung của subspace patching. Makelov et al. (ICLR 2024) chỉ ra rằng patch một subspace có thể kích hoạt một dormant pathway vốn tách rời khỏi tính toán thông thường. Zhang & Nanda (ICLR 2024) cho thấy kết quả định vị thay đổi theo metric và cách tạo corruption.

### 2.4. Causal Tracing trong diffusion ảnh: Basu et al. (ICLR 2024; ICML 2024) [A*] và Causal Representation Editing (NeurIPS 2024) [A*]

**Đã giải quyết được gì.** Basu et al. (ICLR 2024) đưa causal mediation analysis / causal tracing từ LLM sang Stable Diffusion. Họ phát hiện rằng trong CLIP text encoder chỉ có *một* causal state duy nhất cho mỗi thuộc tính thị giác, nằm ở self-attention layer đầu tiên tại token chủ thể cuối cùng, còn trong UNet thì tri thức phân tán. Từ đó họ xây dựng DiffQuickFix, một phép chỉnh sửa dạng closed-form để xóa khái niệm. Ở ICML 2024, cùng nhóm cho thấy causal tracing cổ điển *thất bại* trên SD-XL và DeepFloyd, và đề xuất LocoGen đo direct effect của can thiệp lên cross-attention. Theo hướng an toàn, một bài NeurIPS 2024 về causal representation editing chỉ can thiệp tại các *timestep có liên hệ nhân quả* với khái niệm độc hại, được xác định bằng thủ tục "assess-with-exclusion". Đây là một trong số ít công trình định vị nhân quả theo trục timestep cho mục đích an toàn.

**Chưa giải quyết được gì.** Các kết quả này dành cho diffusion liên tục trong không gian latent ảnh. Ở đó điều kiện được mã hóa một lần bởi text encoder, và không có cơ chế *commit token rời rạc, không thể đảo ngược* như masked dLLM. Vì vậy, chúng không mô hình hóa được hiện tượng path dependence đặc trưng của dLLM: can thiệp tại bước $k$ làm thay đổi tập token được unmask ở các bước sau, và hiệu ứng tích lũy qua cả quỹ đạo. Chính Basu et al. cũng chỉ ra rằng phương pháp định vị không chuyển được giữa các kiến trúc. Do đó, việc chuyển causal tracing theo timestep sang dLLM văn bản, cho hành vi refusal, là một bài toán mới chứ không phải phép áp dụng trực tiếp.

### 2.5. DiffuGuard: Li et al., *How Intrinsic Safety is Lost and Found in Diffusion Large Language Models* (ICLR 2026) [A*]

**Đã giải quyết được gì.** DiffuGuard là một trong những công trình đầu tiên phân tích *vì sao* dLLM mất an toàn trong quá trình denoising, gọi là "denoising-path dependence". Các thí nghiệm can thiệp trên output cho thấy ép token đầu thành "Sure" làm ASR tăng 76,9%, còn ép thành "Sorry" làm ASR giảm 24,3%. Chèn "Sorry" càng sớm thì hiệu quả càng cao. Tác giả cũng cho thấy chiến lược remasking theo low-confidence (greedy) có xu hướng loại bỏ token an toàn để giữ token độc hại có confidence cao. Dựa trên đó, họ đề xuất một defense training-free gồm hai thành phần: Stochastic Annealing Remasking, và Block-level Audit & Repair dùng "Safety Divergence" (cosine distance giữa hidden state của yêu cầu lõi và của toàn bộ prompt jailbreak). ASR trung bình trên sáu loại tấn công và bốn dLLM giảm từ 47,9% xuống 14,7% khi kết hợp với Self-reminder, và xuống 27,8% khi dùng DiffuGuard một mình.

**Chưa giải quyết được gì.** Bằng chứng nhân quả của DiffuGuard dừng ở mức *token output* (ép "Sure"/"Sorry"), không chỉ ra trạng thái nội tại nào (layer, bước, vị trí) mang quyết định an toàn. Safety Divergence là một heuristic hình học dựa trên tương quan, cần biết trước "yêu cầu lõi" và dùng ngưỡng cố định $\lambda=0{,}1$. Defense cũng luôn can thiệp ở block đầu tiên, thay vì tại nơi được chứng minh là có ảnh hưởng nhân quả. Hệ quả là dưới adaptive attack dò ngưỡng, ASR trên Dream vẫn đạt 84,62%. Các tấn công dùng để đánh giá chủ yếu được thiết kế cho AR, và threat model không bao gồm can thiệp vào quỹ đạo denoising.

### 2.6. Beyond Token Positions / RAEC: Wang et al., *Safety Alignment Across Denoising Steps in Diffusion Language Models* (EMNLP 2026) [A*, theo arXiv]

**Đã giải quyết được gì.** Công trình này mở rộng "shallow safety alignment" của Qi et al. sang dLLM và đặt tên hiện tượng là "shallow-step alignment": các commitment ở những bước denoising đầu quyết định an toàn, kể cả ở vị trí không nằm ở đầu chuỗi. Trên LLaDA-Instruct, chỉ một compliance prefill ở vị trí 1% đã đẩy ASR từ 1,29% lên 93,93%. Ép một refusal token sau khoảng bước 8 hầu như không còn tác dụng trên base model. Phát hiện tinh tế nhất là tín hiệu refusal *xuất hiện* ở 98% quỹ đạo không an toàn. Điều phân biệt quỹ đạo an toàn với không an toàn là *tính bền vững* của refusal: quỹ đạo an toàn commit refusal token 98% số lần (so với 4,10%), và duy trì lâu hơn (21,28 so với 7,41 bước). Từ đó tác giả đề xuất RAEC (Refusal-Aware Early Commitment), một decoding rule training-free gần như không làm giảm utility.

**Chưa giải quyết được gì.** Mọi can thiệp trong công trình đều trên *token output* (prefill, ép token) và mọi thống kê đều quan sát xác suất token. Không có activation patching hay probing nào được thực hiện. Vì vậy câu hỏi then chốt mà chính phát hiện của họ đặt ra vẫn chưa có lời giải: *trạng thái nội tại nào trung gian (mediate) cho tính bền vững của refusal*, tức là vì sao cùng một tín hiệu refusal xuất hiện nhưng ở quỹ đạo này thì được giữ, ở quỹ đạo kia thì bị ghi đè. Hiệu quả của RAEC còn khiêm tốn trên LLaDA (DIJA-HarmBench 40,5 → 34,0). Công trình chỉ xét kiến trúc vanilla, chưa xét động lực khi fine-tuning, và chưa so sánh trực tiếp với A2D, DiffuGuard hay MOSA.

### 2.7. TrajHijack: Singh, *Re-Mask and Redirect: Exploiting Denoising Irreversibility in Diffusion Language Models* (2026) [arXiv]

**Đã giải quyết được gì.** TrajHijack xác lập một threat model mới dành riêng cho dLLM. Kẻ tấn công white-box không sửa prompt hay trọng số, mà chỉ re-mask các refusal token đã được commit rồi chèn một affirmative prefix. Tấn công đạt 74–82% ASR trên HarmBench với ba dLLM đã được safety-tune, và 92–98% với prefix 8 token. Công trình cung cấp bằng chứng định lượng về "Early Commitment Effect": refusal thường được commit trước bước 8 trên 64. Tuy vậy, re-mask ở bất kỳ đâu từ bước 4 đến bước 48 vẫn cho 82–92% ASR, nghĩa là commit sớm là *cần* nhưng *không bất khả đảo*. Đáng chú ý nhất là "Defense Inversion Effect": model đã align bằng A2D lại *dễ tổn thương hơn* (89,9%) so với model không có defense (76,1%).

**Chưa giải quyết được gì.** TrajHijack chứng minh rằng an toàn của dLLM đang "nằm" ở các token đã commit chứ không nằm trong một biểu diễn nội tại bền vững. Tuy nhiên, công trình không chỉ ra biểu diễn nội tại nào *lẽ ra* phải mang quyết định an toàn, hay vì sao các trạng thái ẩn sau khi re-mask không tự khôi phục được refusal. Thí nghiệm chỉ dùng greedy decoding với lịch unmask tuyến tính, trên hai đến ba model, với ablation n=50 (khoảng tin cậy ±10 điểm phần trăm), và ASR giảm mạnh khi độ dài sinh tăng. Các defense được gợi ý, như kiểm tra mask count đơn điệu hay re-verify token đã commit, vẫn là quy tắc ở mức output, chưa dựa trên hiểu biết về cơ chế.

### 2.8. A2D: Jeung et al., *Any-Order, Any-Step Safety Alignment for Diffusion Language Models* (ICLR 2026) [A*]

**Đã giải quyết được gì.** A2D là defense dạng training mạnh nhất hiện nay cho dLLM. Mô hình được huấn luyện, qua LoRA trên LLaDA, LLaDA-1.5 và Dream, để sinh [EOS] tại *bất kỳ* vị trí bị mask nào mà nội dung độc hại sắp xuất hiện, bất kể thứ tự hay bước sinh. ASR của DIJA giảm từ khoảng 83% xuống 0–3,5%, với tỉ lệ false positive 0% trên XSTest và utility gần như được giữ nguyên. Xác suất [EOS] ở bước 1 tại vị trí mask ngoài cùng bên trái đủ dự báo để từ chối sớm, nhanh hơn tới 19,3 lần.

**Chưa giải quyết được gì.** A2D giám sát ở mức token output và đối xử đồng nhất với mọi vị trí và mọi bước. Nó không dựa trên, và cũng không tạo ra, hiểu biết về trạng thái nội tại nào thực sự quyết định an toàn. Vì cơ chế an toàn được "dồn" vào việc commit [EOS] sớm, A2D trở thành điểm hỏng duy nhất khi kẻ tấn công can thiệp được vào quỹ đạo: dưới TrajHijack, model được A2D bảo vệ có ASR 89,9%, cao hơn model gốc. Đánh giá adaptive attack của A2D chỉ giới hạn ở các prompt do chính tác giả thiết kế.

### 2.9. Safety Neurons trong dLLM: Dumitrescu et al., *Diffusion LLMs as Targets and Adversaries: Mechanistic Safety Exploits* (2026) [arXiv]

**Đã giải quyết được gì.** Đây là công trình mechanistic đầu tiên về safety trong dLLM. Tác giả xác định các FFN safety neuron bằng một probe logistic regression phân biệt prompt lành tính và độc hại, giữ lại 0,8–3% neuron hàng đầu. Pruning các neuron này đẩy ASR của LLaDA từ 2,6% lên 73,8% và của Dream từ 1,9% lên 86,6%, cho thấy an toàn của dLLM thưa và mong manh. Công trình còn cho thấy an toàn được *kế thừa* từ AR model gốc: Fast-dLLM chia sẻ khoảng 41,8% và Dream khoảng 30,9% safety neuron với Qwen2.5, trong khi LLaDA (huấn luyện từ đầu) có 0% overlap với Llama. Transfer pruning từ Qwen2.5 cũng phá được Dream (ASR 73,2%).

**Chưa giải quyết được gì.** Neuron được xác định *chỉ ở bước encode prompt*; toàn bộ các bước denoising sau đó bị bỏ qua. Như vậy công trình định vị được "ở đâu" (neuron) nhưng không định vị được "khi nào" (bước) và "tại vị trí phản hồi nào". Việc chọn neuron dựa trên probe nên mang tính tương quan, và pruning chỉ đo tính *cần*, không kiểm tra tính *đủ* (patch các neuron này vào quỹ đạo jailbreak có khôi phục refusal hay không). Kết quả dựa vào automatic judge, transfer đòi hỏi kiến trúc giống hệt nhau, và tác giả thừa nhận "Jailbreak Zone" có thể là artifact của dataset.

### 2.10. Mechanistic Interpretability tổng quát cho dLLM: DLM-Scope (Wang et al., 2026) [arXiv] và Steering Without Breaking (Zhou et al., 2026) [arXiv]

**Đã giải quyết được gì.** DLM-Scope huấn luyện Top-K SAE ở sáu layer residual của LLaDA-8B và Dream-7B, với các biến thể Mask-SAE và Unmask-SAE. Kết quả cho thấy steering *trong quá trình denoising* thường hiệu quả hơn steering một lần như ở AR, nhất là ở layer sâu, và SAE của base model chuyển được sang instruct model, trừ ở layer sâu nhất. Steering Without Breaking dùng SAE, một "diffusion logit lens" và theo dõi feature theo từng bước trên MDLM, SEDD, LLaDA và Dream. Công trình cho thấy các thuộc tính ngữ nghĩa *hình thành ở những bước khác nhau*: LLaDA hình thành dần, còn Dream (khởi tạo từ AR) dồn sự hình thành vào một phần tư cuối của quỹ đạo, và không có lịch steering cố định nào tối ưu. Hai công trình này cung cấp bộ công cụ và bằng chứng ban đầu rằng trục thời gian denoising có cấu trúc biểu diễn riêng.

**Chưa giải quyết được gì.** Cả hai đều *không đánh giá safety*, và bằng chứng về thứ tự hình thành hay thứ tự decoding *chỉ mang tính tương quan*. Một công trình khác (*Answer First, Reason Later*, 2026, arXiv) còn cảnh báo rằng thứ tự commit token nhìn thấy được không phản ánh thứ tự tính toán latent. Như vậy, quan sát "token refusal được commit ở bước 8" chưa nói được quyết định từ chối thực sự được *tính* ở bước nào. Việc Dream và LLaDA có động lực hình thành khác nhau cũng đặt ra yêu cầu rằng mọi kết luận định vị phải được kiểm tra riêng cho từng họ model.

---

## 3. Tổng hợp khoảng trống

Gộp các công trình trên lại, có bốn khoảng trống đan xen nhau:

- **(G1) Thiếu trục thời gian trong định vị nhân quả.** Định vị nhân quả cho AR (Arditi, LOCA, Ships, Safety Layers) chỉ hoạt động trên lưới layer × vị trí của một forward pass và đo ở token đầu tiên. Định vị trong dLLM (Dumitrescu et al.) chỉ xét bước encode prompt. Không công trình nào patch hidden state *theo từng bước denoising*.
- **(G2) Bằng chứng theo bước chỉ ở mức output.** Các bằng chứng "early-step" trong dLLM (DiffuGuard, RAEC, TrajHijack) đều là can thiệp trên *token output*, chưa phải trên *trạng thái nội tại*. Vì thế chúng không phân biệt được bước model *ra quyết định* với bước token được *commit*.
- **(G3) Path dependence chưa được định nghĩa.** Can thiệp vào trạng thái ở bước $k$ còn thay đổi tập vị trí được unmask ở các bước sau. Do đó khái niệm hiệu ứng trực tiếp và hiệu ứng toàn phần của một trạng thái lên kết quả an toàn cuối cùng chưa được định nghĩa cho dLLM. Bài toán này không có tương đương trong patching một forward pass hay trong diffusion ảnh liên tục.
- **(G4) Các rủi ro phương pháp luận chưa được kiểm soát trong bối cảnh dLLM.** Gồm subspace-patching illusion (Makelov et al., ICLR 2024), độ nhạy theo metric và corruption (Zhang & Nanda, ICLR 2024), tính không khả định danh (non-identifiability) của giải thích cơ chế (Méloux et al., ICLR 2025), và proxy token đầu. Thêm vào đó, các defense hiện tại (A2D, DiffuGuard, RAEC) đều đặt cơ chế an toàn vào việc commit sớm ở mức output, và TrajHijack cho thấy đây là điểm hỏng duy nhất.

---

## 4. Phát biểu bài toán

Xét một masked dLLM $f_\theta$ có $L$ layer, sinh phản hồi độ dài $N$ trong $K$ bước denoising. Ở bước $k \in \{1,\dots,K\}$, trạng thái chuỗi là $z_k$ (gồm các token đã commit và các [MASK]). Hidden state tại layer $\ell$, vị trí $i$ là $h^{(\ell)}_{k,i}$. Chính sách unmask $\pi$ chọn tập vị trí $U_k$ để commit ở bước $k$. Đầu ra cuối là $y = z_K$, và kết quả an toàn $H(y) \in [0,1]$ được đo trên *toàn bộ văn bản cuối* (ví dụ HarmBench classifier, StrongREJECT) chứ không phải trên token đầu.

Cho một cặp quỹ đạo: quỹ đạo *clean* $\tau^{+}$ (model từ chối) và quỹ đạo *corrupt* $\tau^{-}$ (model bị jailbreak, ví dụ bằng DIJA, PAD hoặc TrajHijack). Hai quỹ đạo dùng cùng yêu cầu lõi và cùng seed. Một can thiệp $\mathrm{do}(h^{(\ell)}_{k,i} \leftarrow \tilde h)$ thay trạng thái của $\tau^{-}$ bằng trạng thái tương ứng của $\tau^{+}$ (hoặc ngược lại). Có hai chế độ:

- **Total effect (TE):** cho phép chính sách unmask $\pi$ phản ứng tự do với can thiệp, nên hiệu ứng lan qua cả biểu diễn lẫn các token được commit về sau.
- **Controlled direct effect (CDE):** cố định thứ tự unmask $\{U_{k'}\}_{k'>k}$ theo quỹ đạo gốc, nên chỉ đo hiệu ứng truyền qua biểu diễn.

Một tập trạng thái $S \subseteq \{(\ell,k,i)\}$ được gọi là **safety-critical** nếu nó vừa:

- **đủ (sufficient):** patch $S$ từ $\tau^{+}$ vào $\tau^{-}$ khôi phục refusal, tức $H(y)$ giảm dưới ngưỡng $\epsilon_s$;
- **cần (necessary):** ablate hoặc patch $S$ từ $\tau^{-}$ vào $\tau^{+}$ làm mất refusal, tức $H(y)$ vượt ngưỡng $\epsilon_n$;

đồng thời $|S|$ là nhỏ nhất. Đồng thời, việc patch không làm tăng over-refusal trên XSTest và không làm suy giảm utility.

---

## 5. Câu hỏi nghiên cứu

**RQ1. Định vị theo không gian–thời gian.** Trong các masked dLLM (LLaDA-8B-Instruct, LLaDA-1.5, Dream-7B), có tồn tại một tập nhỏ các trạng thái nội tại $(\ell, k, i)$ vừa *cần* vừa *đủ* cho hành vi từ chối, khi kết quả được đo trên văn bản cuối thay vì token đầu hay không? Nếu có, tập này tập trung ở những layer nào, bước denoising nào, và tại vị trí prompt (template, dấu câu) hay vị trí phản hồi?

**RQ2. Tách bước "ra quyết định" khỏi bước "commit" và phân rã path dependence.** Hiệu ứng nhân quả của các trạng thái ở bước sớm lên an toàn cuối cùng đi theo kênh nào? Có hai kênh: kênh *trực tiếp* qua biểu diễn (CDE với thứ tự unmask cố định), và kênh *gián tiếp* qua các token được commit (TE − CDE). Bước mà quyết định an toàn được *tính* trong biểu diễn có đi trước bước mà refusal token được *commit* hay không?

**RQ3. Trung gian của tính bền vững refusal.** Phát hiện của RAEC cho thấy tín hiệu refusal xuất hiện ở cả quỹ đạo an toàn lẫn không an toàn, chỉ khác ở tính bền vững. Vậy có trạng thái nội tại cụ thể nào *trung gian* cho sự khác biệt này không? Trạng thái đó có giải thích được vì sao TrajHijack (re-mask từ bước 4 đến 48) và Defense Inversion Effect trên A2D lại thành công không?

**RQ4. Hình học refusal theo thời gian và sự kế thừa từ AR.** dLLM có một hướng refusal hay một refusal cone không, và hình học đó *biến đổi thế nào qua các bước denoising* (xoay, dịch chuyển hay xuất hiện muộn)? Các trạng thái safety-critical của Dream (khởi tạo từ Qwen2.5) có trùng với safety head, safety layer hay refusal direction của AR model gốc không, và chúng khác LLaDA (huấn luyện từ đầu) ra sao?

**RQ5. Độ tin cậy của phép định danh.** Tập trạng thái tìm được có *ổn định* qua các điều kiện sau không:

- các họ tấn công khác nhau (DIJA, PAD, MaskForge, TrajHijack);
- các sampler và lịch unmask khác nhau (greedy/stochastic, low-confidence/random);
- các phương pháp định vị khác nhau (activation patching, attribution patching, SAE-direction patching, DAS).

Tập đó có vượt qua được các kiểm tra chống illusion và non-identifiability không?

**RQ6 (cầu nối sang can thiệp thích ứng).** Điểm hiệu ứng nhân quả ước lượng trực tuyến (ví dụ bằng attribution patching) có thể dùng làm tín hiệu quyết định *khi nào* (bước nào) và *ở đâu* (layer, vị trí nào) cần can thiệp không? So với các gate dựa trên probe tương quan (CAST, SafeSwitch, AdaSteer, DiffuGuard), tín hiệu này có cho trade-off tốt hơn giữa ASR, over-refusal và utility, và có bền vững hơn trước adaptive attack không?
