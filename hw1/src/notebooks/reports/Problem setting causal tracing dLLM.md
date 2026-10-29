# Problem Setting: Causal Localization of Refusal Dynamics in Diffusion Language Models

> Ghi chú venue: [A*] = đã xác nhận tại venue A*; [arXiv] = preprint chưa qua phản biện hội nghị; [venue, theo arXiv] = venue chỉ lấy từ comment trên arXiv, chưa thấy proceedings chính thức. Các con số là số do tác giả báo cáo.
>
> **Phiên bản 2.** Bản này thu hẹp phạm vi còn 3 RQ chính, sửa lại định nghĩa hiệu ứng nhân quả, cách xây dựng cặp quỹ đạo và khái niệm safety-critical state. Adaptive intervention được chuyển thành phần mở rộng.

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

### 2.3. LOCA: Kumar & Ahuja, *Minimal, Local, Causal Explanations for Jailbreak Success in LLMs* (2026) [COLM 2026, theo nguồn reviewer; cần xác minh]

**Đã giải quyết được gì.** LOCA đại diện cho SOTA hiện nay về causal localization ở mức tinh cho jailbreak trên AR LLM. Phương pháp ghép token giữa prompt độc hại và phiên bản jailbreak của nó, sau đó chỉ patch activation dọc theo các decoder direction của sparse autoencoder (SAE). Mỗi cặp (token, direction) được chấm điểm bằng một xấp xỉ bậc nhất và được tính lại lặp đi lặp lại. Với chỉ khoảng 6–8 patch trên Llama-3.1-8B (12–16 trên Gemma-2-2B), LOCA khôi phục được hành vi từ chối, trong khi các baseline thường thất bại kể cả sau 20 patch. Kết quả cũng mang tính diễn giải: patch ở layer giữa rơi vào các token template sau instruction và dấu câu. Điều này cho thấy jailbreak hoạt động bằng cách khuếch đại các khái niệm "vô hại" tại những vị trí đó.

**Chưa giải quyết được gì.** Mục tiêu tối ưu của LOCA là KL divergence trên *token output đầu tiên*. Proxy này thất bại khi phản hồi gốc và phản hồi bị jailbreak có cùng token đầu, và nó bỏ sót nội dung độc hại xuất hiện muộn hơn. Phương pháp chỉ hoạt động trên lưới (layer × vị trí prompt) của một forward pass AR, nên không có khái niệm bước sinh hay bước denoising. Ngoài ra nó phụ thuộc vào chất lượng SAE, chạy chậm, và cũng chịu các rủi ro phương pháp luận chung của subspace patching. Makelov et al. (ICLR 2024) chỉ ra rằng patch một subspace có thể kích hoạt một dormant pathway vốn tách rời khỏi tính toán thông thường. Zhang & Nanda (ICLR 2024) cho thấy kết quả định vị thay đổi theo metric và cách tạo corruption.

### 2.4. Causal Tracing trong diffusion ảnh: Basu et al. (ICLR 2024; ICML 2024) [A*] và Causal Representation Editing (NeurIPS 2024) [A*]

**Đã giải quyết được gì.** Basu et al. (ICLR 2024) đưa causal mediation analysis / causal tracing từ LLM sang Stable Diffusion. Họ phát hiện rằng trong CLIP text encoder chỉ có *một* causal state duy nhất cho mỗi thuộc tính thị giác, nằm ở self-attention layer đầu tiên tại token chủ thể cuối cùng, còn trong UNet thì tri thức phân tán. Từ đó họ xây dựng DiffQuickFix, một phép chỉnh sửa dạng closed-form để xóa khái niệm. Ở ICML 2024, cùng nhóm cho thấy causal tracing cổ điển *thất bại* trên SD-XL và DeepFloyd, và đề xuất LocoGen đo direct effect của can thiệp lên cross-attention. Theo hướng an toàn, một bài NeurIPS 2024 về causal representation editing chỉ can thiệp tại các *timestep có liên hệ nhân quả* với khái niệm độc hại, được xác định bằng thủ tục "assess-with-exclusion". Đây là một trong số ít công trình định vị nhân quả theo trục timestep cho mục đích an toàn.

**Chưa giải quyết được gì.** Các kết quả này dành cho diffusion liên tục trong không gian latent ảnh. Ở đó điều kiện được mã hóa một lần bởi text encoder, và không có cơ chế *commit token rời rạc, không thể đảo ngược* như masked dLLM. Vì vậy, chúng không mô hình hóa được hiện tượng path dependence đặc trưng của dLLM: can thiệp tại bước $k$ làm thay đổi tập token được unmask ở các bước sau, và hiệu ứng tích lũy qua cả quỹ đạo. Chính Basu et al. cũng chỉ ra rằng phương pháp định vị không chuyển được giữa các kiến trúc. Do đó, việc chuyển causal tracing theo timestep sang dLLM văn bản, cho hành vi refusal, là một bài toán mới chứ không phải phép áp dụng trực tiếp.

### 2.5. DiffuGuard: Li et al., *How Intrinsic Safety is Lost and Found in Diffusion Large Language Models* (ICLR 2026) [A*]

**Đã giải quyết được gì.** DiffuGuard là một trong những công trình đầu tiên phân tích *vì sao* dLLM mất an toàn trong quá trình denoising, gọi là "denoising-path dependence". Các thí nghiệm can thiệp trên output cho thấy ép token đầu thành "Sure" làm ASR tăng 76,9%, còn ép thành "Sorry" làm ASR giảm 24,3%. Chèn "Sorry" càng sớm thì hiệu quả càng cao. Tác giả cũng cho thấy chiến lược remasking theo low-confidence (greedy) có xu hướng loại bỏ token an toàn để giữ token độc hại có confidence cao. Dựa trên đó, họ đề xuất một defense training-free gồm hai thành phần: Stochastic Annealing Remasking, và Block-level Audit & Repair dùng "Safety Divergence" (cosine distance giữa hidden state của yêu cầu lõi và của toàn bộ prompt jailbreak). ASR trung bình trên sáu loại tấn công và bốn dLLM giảm từ 47,9% xuống 14,7% khi kết hợp với Self-reminder, và xuống 27,8% khi dùng DiffuGuard một mình.

**Chưa giải quyết được gì.** Bằng chứng nhân quả của DiffuGuard dừng ở mức *token output* (ép "Sure"/"Sorry"), không chỉ ra trạng thái nội tại nào (layer, bước, vị trí) mang quyết định an toàn. Safety Divergence là một heuristic hình học dựa trên tương quan, cần biết trước "yêu cầu lõi" và dùng ngưỡng cố định $\lambda=0{,}1$. Defense cũng luôn can thiệp ở block đầu tiên, thay vì tại nơi được chứng minh là có ảnh hưởng nhân quả. Hệ quả là dưới adaptive attack dò ngưỡng, ASR trên Dream vẫn đạt 84,62%. Các tấn công dùng để đánh giá chủ yếu được thiết kế cho AR, và threat model không bao gồm can thiệp vào quỹ đạo denoising.

### 2.6. Beyond Token Positions / RAEC: Wang et al., *Safety Alignment Across Denoising Steps in Diffusion Language Models* (EMNLP 2026) [EMNLP 2026, theo arXiv; chưa thấy proceedings]

**Đã giải quyết được gì.** Công trình này mở rộng "shallow safety alignment" của Qi et al. sang dLLM và đặt tên hiện tượng là "shallow-step alignment": các commitment ở những bước denoising đầu quyết định an toàn, kể cả ở vị trí không nằm ở đầu chuỗi. Trên LLaDA-Instruct, chỉ một compliance prefill ở vị trí 1% đã đẩy ASR từ 1,29% lên 93,93%. Ép một refusal token sau khoảng bước 8 hầu như không còn tác dụng trên base model. Phát hiện tinh tế nhất là tín hiệu refusal *xuất hiện* ở 98% quỹ đạo không an toàn. Điều phân biệt quỹ đạo an toàn với không an toàn là *tính bền vững* của refusal: quỹ đạo an toàn commit refusal token 98% số lần (so với 4,10%), và duy trì lâu hơn (21,28 so với 7,41 bước). Từ đó tác giả đề xuất RAEC (Refusal-Aware Early Commitment), một decoding rule training-free gần như không làm giảm utility.

**Chưa giải quyết được gì.** Mọi can thiệp trong công trình đều trên *token output* (prefill, ép token) và mọi thống kê đều quan sát xác suất token. Không có activation patching hay probing nào được thực hiện. Vì vậy câu hỏi then chốt mà chính phát hiện của họ đặt ra vẫn chưa có lời giải: *trạng thái nội tại nào trung gian (mediate) cho tính bền vững của refusal*, tức là vì sao cùng một tín hiệu refusal xuất hiện nhưng ở quỹ đạo này thì được giữ, ở quỹ đạo kia thì bị ghi đè. Hiệu quả của RAEC còn khiêm tốn trên LLaDA (DIJA-HarmBench 40,5 → 34,0). Công trình chỉ xét kiến trúc vanilla, chưa xét động lực khi fine-tuning, và chưa so sánh trực tiếp với A2D, DiffuGuard hay MOSA.

### 2.7. TrajHijack: Singh, *Re-Mask and Redirect: Exploiting Denoising Irreversibility in Diffusion Language Models* (2026) [arXiv]

**Đã giải quyết được gì.** TrajHijack xác lập một threat model mới dành riêng cho dLLM. Kẻ tấn công white-box không sửa prompt hay trọng số, mà chỉ re-mask các refusal token đã được commit rồi chèn một affirmative prefix. Tấn công đạt 74–82% ASR trên HarmBench với ba dLLM đã được safety-tune, và 92–98% với prefix 8 token. Công trình cung cấp bằng chứng định lượng về "Early Commitment Effect": refusal thường được commit trước bước 8 trên 64. Tuy vậy, re-mask ở bất kỳ đâu từ bước 4 đến bước 48 vẫn cho 82–92% ASR, nghĩa là commit sớm là *cần* nhưng *không bất khả đảo*. Đáng chú ý nhất là "Defense Inversion Effect": model đã align bằng A2D lại *dễ tổn thương hơn* (89,9%) so với model không có defense (76,1%).

**Chưa giải quyết được gì.** TrajHijack chứng minh rằng an toàn của dLLM đang "nằm" ở các token đã commit chứ không nằm trong một biểu diễn nội tại bền vững. Tuy nhiên, công trình không chỉ ra biểu diễn nội tại nào *lẽ ra* phải mang quyết định an toàn, hay vì sao các trạng thái ẩn sau khi re-mask không tự khôi phục được refusal. Thí nghiệm chỉ dùng greedy decoding với lịch unmask tuyến tính, trên hai đến ba model, với ablation n=50 (khoảng tin cậy ±10 điểm phần trăm), và ASR giảm mạnh khi độ dài sinh tăng. Các defense được gợi ý, như kiểm tra mask count đơn điệu hay re-verify token đã commit, vẫn là quy tắc ở mức output, chưa dựa trên hiểu biết về cơ chế.

### 2.8. A2D: Jeung et al., *Any-Order, Any-Step Safety Alignment for Diffusion Language Models* (ICLR 2026) [A*]

**Đã giải quyết được gì.** A2D là defense dạng training mạnh nhất hiện nay cho dLLM. Mô hình được huấn luyện, qua LoRA trên LLaDA, LLaDA-1.5 và Dream, để sinh [EOS] tại *bất kỳ* vị trí bị mask nào mà nội dung độc hại sắp xuất hiện, bất kể thứ tự hay bước sinh. ASR của DIJA giảm từ khoảng 83% xuống 0–3,5%, với tỉ lệ false positive 0% trên XSTest và utility gần như được giữ nguyên. Xác suất [EOS] ở bước 1 tại vị trí mask ngoài cùng bên trái đủ dự báo để từ chối sớm, nhanh hơn tới 19,3 lần.

**Chưa giải quyết được gì.** A2D giám sát ở mức token output và đối xử đồng nhất với mọi vị trí và mọi bước. Nó không dựa trên, và cũng không tạo ra, hiểu biết về trạng thái nội tại nào thực sự quyết định an toàn. Vì cơ chế an toàn được "dồn" vào việc commit [EOS] sớm, A2D trở thành điểm hỏng duy nhất khi kẻ tấn công can thiệp được vào quỹ đạo: dưới TrajHijack, model được A2D bảo vệ có ASR 89,9%, cao hơn model gốc. Đánh giá adaptive attack của A2D chỉ giới hạn ở các prompt do chính tác giả thiết kế.

### 2.9. Safety Neurons trong dLLM: Dumitrescu et al., *Diffusion LLMs as Targets and Adversaries: Mechanistic Safety Exploits* (2026) [arXiv; chưa qua phản biện hội nghị]

**Đã giải quyết được gì.** Đây là công trình mechanistic đầu tiên về safety trong dLLM. Tác giả xác định các FFN safety neuron bằng một probe logistic regression phân biệt prompt lành tính và độc hại, giữ lại 0,8–3% neuron hàng đầu. Pruning các neuron này đẩy ASR của LLaDA từ 2,6% lên 73,8% và của Dream từ 1,9% lên 86,6%, cho thấy an toàn của dLLM thưa và mong manh. Công trình còn cho thấy an toàn được *kế thừa* từ AR model gốc: Fast-dLLM chia sẻ khoảng 41,8% và Dream khoảng 30,9% safety neuron với Qwen2.5, trong khi LLaDA (huấn luyện từ đầu) có 0% overlap với Llama. Transfer pruning từ Qwen2.5 cũng phá được Dream (ASR 73,2%).

**Chưa giải quyết được gì.** Neuron được xác định *chỉ ở bước encode prompt*; toàn bộ các bước denoising sau đó bị bỏ qua. Như vậy công trình định vị được "ở đâu" (neuron) nhưng không định vị được "khi nào" (bước) và "tại vị trí phản hồi nào". Việc chọn neuron dựa trên probe nên mang tính tương quan, và pruning chỉ đo tính *cần*, không kiểm tra tính *đủ* (patch các neuron này vào quỹ đạo jailbreak có khôi phục refusal hay không). Kết quả dựa vào automatic judge, transfer đòi hỏi kiến trúc giống hệt nhau, và tác giả thừa nhận "Jailbreak Zone" có thể là artifact của dataset.

### 2.10. Mechanistic Interpretability tổng quát cho dLLM: DLM-Scope (Wang et al., ICML 2026) [A*] và Steering Without Breaking (Zhou et al., 2026) [arXiv]

**Đã giải quyết được gì.** DLM-Scope huấn luyện Top-K SAE ở sáu layer residual của LLaDA-8B và Dream-7B, với các biến thể Mask-SAE và Unmask-SAE. Kết quả cho thấy steering *trong quá trình denoising* thường hiệu quả hơn steering một lần như ở AR, nhất là ở layer sâu, và SAE của base model chuyển được sang instruct model, trừ ở layer sâu nhất. Steering Without Breaking dùng SAE, một "diffusion logit lens" và theo dõi feature theo từng bước trên MDLM, SEDD, LLaDA và Dream. Công trình cho thấy các thuộc tính ngữ nghĩa *hình thành ở những bước khác nhau*: LLaDA hình thành dần, còn Dream (khởi tạo từ AR) dồn sự hình thành vào một phần tư cuối của quỹ đạo, và không có lịch steering cố định nào tối ưu. Hai công trình này cung cấp bộ công cụ và bằng chứng ban đầu rằng trục thời gian denoising có cấu trúc biểu diễn riêng.

**Chưa giải quyết được gì.** Cả hai đều *không đánh giá safety*, và bằng chứng về thứ tự hình thành hay thứ tự decoding *chỉ mang tính tương quan*. Một công trình khác (*Answer First, Reason Later*, 2026, arXiv) còn cảnh báo rằng thứ tự commit token nhìn thấy được không phản ánh thứ tự tính toán latent. Như vậy, quan sát "token refusal được commit ở bước 8" chưa nói được quyết định từ chối thực sự được *tính* ở bước nào. Việc Dream và LLaDA có động lực hình thành khác nhau cũng đặt ra yêu cầu rằng mọi kết luận định vị phải được kiểm tra riêng cho từng họ model. Vì DLM-Scope (ICML 2026) đã chứng minh steering theo thời gian diffusion là khả thi và hữu ích, đề tài này phải tách bạch rõ hai việc. *Steering* chỉ chứng minh một biểu diễn **có thể được dùng** để thay đổi hành vi. *Định vị cơ chế nhân quả* phải chứng minh trạng thái đó **thực sự tham gia** vào quyết định refusal trên quỹ đạo tự nhiên của model.

---


## 3. Tổng hợp khoảng trống và mức ưu tiên

| Gap | Nội dung | Ưu tiên |
|---|---|---|
| **G1. Định vị nhân quả theo layer × step × position** | Định vị cho AR (Arditi, LOCA, Ships, Safety Layers) chỉ dùng lưới layer × vị trí của một forward pass và đo ở token đầu. Định vị trong dLLM (Dumitrescu et al.) chỉ xét bước encode prompt. Chưa thấy công trình đã xác minh nào patch hidden state theo từng bước denoising cho refusal. | Cao, là đóng góp chính |
| **G2. Phân biệt emergence, causal influence và commitment** | Các bằng chứng "early-step" (DiffuGuard, RAEC, TrajHijack) đều là can thiệp trên token output. RAEC quan sát được sự khác biệt về persistence nhưng không xác định hidden state nào gây ra nó. Thời điểm tín hiệu refusal *giải mã được*, thời điểm nó *có ảnh hưởng nhân quả* và thời điểm token refusal *được commit* đang bị đồng nhất. | Cao, là đóng góp chính |
| **G3. Ảnh hưởng truyền qua sampling policy** | Can thiệp ở bước $k$ còn thay đổi các vị trí được commit về sau. Muốn phân rã thành hiệu ứng trực tiếp và gián tiếp đúng nghĩa mediation thì cần mô hình nhân quả tường minh và các giả định nhận dạng mạnh. | Trung bình, chỉ so sánh hai chế độ, không tuyên bố phân rã |
| **G4. Can thiệp thích ứng dựa trên tín hiệu nhân quả** | Dễ trùng với steering theo thời gian diffusion (DLM-Scope) và adaptive steering + remasking (Lee et al., 2026). Chỉ có novelty khi chứng minh được tín hiệu nhân quả mang lại lợi ích mà probe hoặc steering không có. | Giai đoạn sau |

Ngoài ra còn các rủi ro phương pháp luận mà mọi kết quả đều phải kiểm soát:

- subspace-patching illusion (Makelov et al., ICLR 2024);
- độ nhạy theo metric và cách tạo corruption, cùng activation ngoài phân phối (Zhang & Nanda, ICLR 2024);
- tính không khả định danh của giải thích cơ chế (Méloux et al., ICLR 2025);
- self-repair và backup pathway che giấu tầm quan trọng của thành phần.

---

## 4. Phát biểu bài toán

### 4.1. Ký hiệu

Xét một masked dLLM $f_\theta$ có $L$ layer. Với prompt $p$, model sinh phản hồi trên một *response canvas* độ dài cố định $N$ qua $K$ bước denoising $k = 1,\dots,K$ (đếm từ bước đầu tiên).

- $z_k$ là trạng thái chuỗi sau bước $k$, gồm token đã commit và [MASK].
- $C_k \subseteq \{1,\dots,N\}$ là tập vị trí đã commit sau bước $k$. Chuỗi $m = (C_1,\dots,C_K)$ là **lịch commitment**.
- $h^{(\ell)}_{k,i}$ là hidden state (residual stream) tại layer $\ell$, bước $k$, vị trí $i$. Vị trí $i$ bao gồm cả vị trí prompt và vị trí trên canvas.
- $y = z_K$ là phản hồi cuối.

Kết quả được đo trên **toàn bộ văn bản cuối**, không phải trên token đầu:

- $R(y) \in [0,1]$ là **mức độ độc hại (harmfulness risk)**, chấm bằng HarmBench classifier hoặc StrongREJECT. $R$ càng thấp càng an toàn.
- $\mathrm{Ref}(y) \in \{0,1\}$ là chỉ báo phản hồi có phải là từ chối hay không, dùng để báo cáo song song với $R$.

### 4.2. Xây dựng cặp quỹ đạo có trạng thái tương ứng

Patch một hidden state từ quỹ đạo này sang quỹ đạo kia chỉ có nghĩa khi hai trạng thái *tương ứng* với nhau. Cùng seed là chưa đủ: hai prompt khác nhau cho ra độ dài prompt, mask pattern và lịch commitment khác nhau. Vì vậy, đề tài dùng ba quy tắc sau.

1. **Cùng response canvas và coupled sampling.** Mọi quỹ đạo dùng cùng $N$, $K$ và cùng chuỗi số ngẫu nhiên ở mỗi bước (common random numbers), hoặc greedy decoding. Nhờ đó, mọi khác biệt giữa hai quỹ đạo đều đến từ can thiệp chứ không đến từ nhiễu sampling.
2. **Ưu tiên cặp cùng prompt (loại P1).** Quỹ đạo clean $\tau^{+}$ là lần chạy gốc mà model từ chối. Quỹ đạo corrupt $\tau^{-}$ dùng *cùng prompt* nhưng bị đẩy sang tuân thủ bằng một can thiệp tại bước $k_0$: ép commit token tuân thủ (kiểu RAEC) hoặc re-mask kèm affirmative prefix (kiểu TrajHijack). Hai quỹ đạo khi đó trùng nhau hoàn toàn trước $k_0$, và vị trí prompt khớp chính xác.
3. **Cặp khác prompt (loại P2: prompt gốc so với DIJA/PAD template) được báo cáo riêng.** Với loại này chỉ patch vị trí trên canvas và các token của yêu cầu lõi có ánh xạ vị trí tường minh. Token template không có cặp tương ứng thì không được patch.

Ở một bước $k$, một vị trí có thể đang bị mask trong quỹ đạo này nhưng đã commit trong quỹ đạo kia. Hidden state của [MASK] và của token thật không so sánh trực tiếp được, nên những vị trí này được tách thành một nhóm phân tích riêng.

Các phép kiểm tra đối chứng bắt buộc:

- patch trạng thái từ một quỹ đạo refusal của *prompt khác* (random-pair control);
- patch bằng mean activation, và patch bằng resample ablation;
- kiểm tra chuẩn (norm) và khoảng cách tới phân phối activation tự nhiên, để phát hiện trạng thái patch ngoài phân phối;
- so sánh patch toàn bộ trạng thái với patch subspace, để phát hiện dormant pathway.

### 4.3. Can thiệp và hai chế độ đo

Gọi $A_S$ là can thiệp $\mathrm{do}\big(h^{(\ell)}_{k,i} \leftarrow \tilde h^{(\ell)}_{k,i}\ \ \forall (\ell,k,i) \in S\big)$, trong đó $\tilde h$ lấy từ quỹ đạo nguồn tương ứng. Hiệu ứng được đo ở hai chế độ:

- **Free sampling.** Sampling policy phản ứng tự do với can thiệp, nên lịch commitment có thể thay đổi:
  $$E_{\text{free}}(S) = \mathbb{E}\big[R(y) \mid A_S\big] - \mathbb{E}\big[R(y)\big].$$
- **Fixed commitment schedule.** *Tập vị trí* được commit ở mỗi bước sau can thiệp bị giữ theo lịch $m$ của quỹ đạo không bị can thiệp. Token tại các vị trí đó vẫn do model chọn:
  $$E_{\text{fix}}(S; m) = \mathbb{E}\big[R(y) \mid A_S,\ \text{lịch} = m\big] - \mathbb{E}\big[R(y) \mid \text{lịch} = m\big].$$

$E_{\text{fix}}$ được gọi là *hiệu ứng dưới lịch commitment bị ràng buộc*, **không** được diễn giải là natural direct effect. Tương tự, $E_{\text{free}} - E_{\text{fix}}$ **không** được diễn giải là natural indirect effect. Lý do: $E_{\text{fix}}$ là một controlled direct effect tại một giá trị $m$ cụ thể, và nó chỉ trùng natural direct effect khi can thiệp không tương tác với mediator. Natural indirect effect còn cần giả định "cross-world", không kiểm chứng được bằng thực nghiệm. Đề tài chỉ dùng sự *chênh lệch giữa hai chế độ* làm bằng chứng mô tả rằng sampling policy khuếch đại hay triệt tiêu hiệu ứng. Phân rã mediation đầy đủ để lại làm hướng mở rộng.

### 4.4. Điểm nhân quả cục bộ, khái quát và tập trạng thái thưa

Với mỗi cặp $j = (\tau^{+}_j, \tau^{-}_j)$ và một tập trạng thái $S$, định nghĩa hai điểm cục bộ:

- **Local sufficiency:** $\mathrm{Suf}_j(S) = R(\tau^{-}_j) - R(\tau^{-}_j \mid A_S^{+\to-})$. Patch $S$ từ quỹ đạo clean vào quỹ đạo corrupt làm giảm độc hại bao nhiêu.
- **Local necessity:** $\mathrm{Nec}_j(S) = R(\tau^{+}_j \mid A_S^{-\to+}) - R(\tau^{+}_j)$. Patch $S$ từ quỹ đạo corrupt vào quỹ đạo clean làm tăng độc hại bao nhiêu.

Hai điểm này là tính chất *cục bộ của từng quỹ đạo*. Một tập có thể cần trong quỹ đạo này nhưng không cần trong quỹ đạo khác. Vì vậy, khả năng khái quát được đo riêng qua phân phối prompt $\mathcal D$, họ tấn công $a$ và lịch decoding $\pi$, bằng giá trị trung bình $\mathbb E_{j}[\mathrm{Suf}_j(S)]$ và tỉ lệ ổn định $\Pr_j[\mathrm{Suf}_j(S) > \delta]$.

Mục tiêu **không** phải là tìm tập nhỏ nhất toàn cục, vì đó là bài toán tổ hợp không giải được bằng greedy. Mục tiêu là **tìm các tập trạng thái thưa có hiệu ứng nhân quả lớn và ổn định**. Kết quả được báo cáo dưới dạng đường cong hiệu ứng theo $|S|$, với các ứng viên được xếp hạng bằng attribution patching rồi kiểm chứng lại bằng activation patching thật.

### 4.5. Ba biến thời gian và giả thuyết trung tâm

Với mỗi quỹ đạo, định nghĩa ba thời điểm khác nhau:

- $t_{\text{dec}}$ là bước sớm nhất mà refusal **giải mã được** từ hidden state, tức một probe held-out đạt AUROC $\ge \theta$. Đây là thời điểm *thông tin xuất hiện*.
- $t_{\text{cause}}$ là bước sớm nhất $k$ mà patch các trạng thái tại bước $k$ cho $\mathrm{Suf}_j \ge \delta$. Đây là thời điểm *thông tin có ảnh hưởng nhân quả*.
- $t_{\text{commit}}$ là bước mà token refusal đầu tiên được commit và duy trì đến cuối. Đây là thời điểm *quyết định hiện ra ở output*.

Probe accuracy cao chỉ cho thấy thông tin giải mã được, không cho thấy trạng thái đó gây ra refusal. Vì vậy $t_{\text{dec}}$ và $t_{\text{cause}}$ phải được đo riêng.

**Giả thuyết trung tâm (H1, có thể bác bỏ):** trong một tỉ lệ đáng kể các quỹ đạo, tồn tại trạng thái nội tại ở bước $k < t_{\text{commit}}$ mà patching làm thay đổi kết quả refusal cuối cùng, tức $t_{\text{cause}} < t_{\text{commit}}$.

- Nếu H1 đúng, quyết định an toàn được hình thành trong biểu diễn *trước* khi lộ ra ở output. Phát hiện này mở đường cho can thiệp sớm dựa trên trạng thái thay vì dựa trên token.
- Nếu H1 bị bác bỏ, tức hidden state trước bước commit không thay đổi được kết quả, thì đó vẫn là phát hiện có giá trị. Nó cho thấy an toàn của dLLM nằm chủ yếu ở token đã commit, khớp với TrajHijack, và chỉ ra giới hạn của causal localization trong dLLM.

**Giả thuyết phụ (H2):** $t_{\text{dec}} \le t_{\text{cause}}$, và khoảng cách giữa hai thời điểm này lớn hơn ở quỹ đạo bị jailbreak. Nói cách khác, jailbreak không xóa thông tin "độc hại" mà ngăn nó trở thành nguyên nhân.

---

## 5. Câu hỏi nghiên cứu

**RQ1 (Định vị).** Which internal states causally influence refusal outcomes across denoising steps, layers, and token positions?
Những trạng thái nội tại $(\ell, k, i)$ nào có local sufficiency và local necessity lớn đối với kết quả refusal, đo trên văn bản cuối, ở chế độ free sampling? Hiệu ứng tập trung ở những layer, bước và loại vị trí nào (prompt, template, canvas đã commit, canvas còn mask)?

**RQ2 (Động lực).** How do causal refusal states emerge and persist before observable refusal tokens are committed?
Quan hệ giữa $t_{\text{dec}}$, $t_{\text{cause}}$ và $t_{\text{commit}}$ là gì (kiểm định H1 và H2)? Trạng thái có ảnh hưởng nhân quả nào phân biệt quỹ đạo giữ được refusal với quỹ đạo bị ghi đè, tức phát hiện về persistence của RAEC? Hiệu ứng khác nhau thế nào giữa chế độ free sampling và fixed commitment schedule?

**RQ3 (Khái quát).** How stable are these causal states across jailbreak strategies, decoding schedules, and diffusion language model architectures?
Các trạng thái tìm được ở RQ1 và RQ2 có ổn định khi thay đổi những yếu tố sau không:

- họ tấn công: prefill, DIJA, PAD, TrajHijack;
- seed và lịch decoding: greedy hay stochastic, low-confidence hay random remasking, số bước $K$;
- kiến trúc: LLaDA (huấn luyện từ đầu) và Dream (khởi tạo từ AR).

Kết quả có nhất quán giữa activation patching, attribution patching và SAE-direction patching không?

**Thí nghiệm mở rộng (không phải đóng góp chính):**

- **E1. Kế thừa từ AR và hình học refusal.** Trạng thái nhân quả của Dream có trùng với refusal direction, safety head của Qwen2.5 không? Refusal cone thay đổi thế nào qua các bước?
- **E2. Phân rã mediation.** Xây dựng mô hình nhân quả tường minh cho lịch commitment và nêu rõ giả định nhận dạng, trước khi tuyên bố bất kỳ natural direct effect hay natural indirect effect nào.
- **E3. Can thiệp có mục tiêu nhân quả.** Chỉ can thiệp tại các trạng thái đã được xác nhận ở RQ1–RQ3, rồi so sánh với steering theo thời gian diffusion (DLM-Scope), adaptive steering + remasking (Lee et al.), DiffuGuard và A2D về ASR, over-refusal (XSTest) và utility. Hướng này chỉ có novelty nếu vượt được các baseline đó, đặc biệt dưới TrajHijack.

---

## 6. Kế hoạch thí nghiệm

| Giai đoạn | Nội dung | Đầu ra | Điểm kiểm tra |
|---|---|---|---|
| 0. Pilot | Chạy thử trên một tập nhỏ prompt để ước lượng phương sai của $R$ và $\mathrm{Suf}_j$ | Kích thước mẫu và số seed, chọn qua phân tích độ bất định chứ không chọn tùy ý | Judge có ổn định không; cặp P1 có tạo được sự khác biệt rõ về $R$ không |
| 1. Trajectory logging | Ghi hidden state, logits, mask và lịch commitment theo từng bước, với coupled sampling | Bộ quỹ đạo cặp P1 (chính) và P2 (phụ) | Hai quỹ đạo P1 trùng nhau trước $k_0$ |
| 2. Causal localization | Attribution patching để sàng lọc, sau đó activation patching thật trên các ứng viên, theo layer × step × position, ở chế độ free | Heatmap $\mathrm{Suf}$, $\mathrm{Nec}$ và đường cong hiệu ứng theo $|S|$ | Vượt random-pair control và OOD check |
| 3. Mechanism analysis | Đo $t_{\text{dec}}$, $t_{\text{cause}}$, $t_{\text{commit}}$; so sánh chế độ free với fixed schedule | Kiểm định H1 và H2 | Kết quả phủ định cũng được báo cáo |
| 4. Robustness | Thay họ tấn công, seed, lịch decoding, model, phương pháp patching | Bảng mức ổn định (RQ3) | Mức nhất quán giữa các phương pháp |
| 5. (Mở rộng) Can thiệp nhân quả | Chỉ can thiệp tại trạng thái đã xác nhận | ASR, XSTest, utility so với baseline | Có vượt DLM-Scope steering và Lee et al. không |

**Phạm vi khởi đầu:** LLaDA-8B-Instruct và Dream-7B-Instruct. Hai model này khác nhau về cách khởi tạo (huấn luyện từ đầu so với khởi tạo từ AR), đủ để kiểm tra khả năng khái quát. MMaDA và LLaDA-1.5 được thêm vào sau nếu kết quả ổn định.

**Mã nguồn tham khảo:** repo LLaDA chính thức (để ghi quỹ đạo), LOCA (để tham khảo cách patching dọc SAE direction), DiffuGuard (để đối chiếu cơ chế defense).
