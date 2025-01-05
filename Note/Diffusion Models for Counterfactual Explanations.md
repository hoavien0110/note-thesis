# Diffusion Models for Counterfactual Explanations

## 1. Intro
Mạng nơ-ron tích chập (Convolutional Neural Networks - CNNs) đã đạt được những kết quả ấn tượng, khó tưởng tượng chỉ vài thập kỷ trước, nhờ vào việc sử dụng các mô hình rất lớn và sâu (ví dụ: với hàng trăm lớp và gần như hàng tỷ tham số có thể huấn luyện). Tuy nhiên, rất khó để giải thích các quyết định của chúng do tính phi tuyến mạnh mẽ và quá mức tham số hóa của chúng. Hơn nữa, trong các ứng dụng thực tế, nếu một mô hình dựa vào các mối tương quan giả (spurious correlations) trong dữ liệu để đưa ra dự đoán, người dùng cuối sẽ nghi ngờ tính hợp lệ của quyết định đó. Đặc biệt, trong các tình huống nhạy cảm như y tế hoặc các hệ thống quan trọng, học máy (Machine Learning - ML) phải đảm bảo sử dụng các đặc trưng đúng đắn để tính toán dự đoán và tránh các mối liên kết giả. Vì lý do này, lĩnh vực nghiên cứu Trí tuệ nhân tạo có thể giải thích được (Explainable Artificial Intelligence - XAI) đã phát triển mạnh mẽ trong những năm gần đây nhằm tiến tới hiểu rõ hơn về cơ chế ra quyết định trong các mô hình hộp đen.

Trong bài báo này, chúng tôi tập trung vào các phương pháp giải thích hậu kỳ (post-hoc explanation methods). Đặc biệt, chúng tôi nhấn mạnh vào nhánh đang phát triển mạnh của các CE (Counterfactual Explanations - CE). CEs nhằm mục đích tạo ra những thay đổi tối thiểu nhưng có ý nghĩa trên một mẫu đầu vào để thay đổi quyết định ban đầu do một mô hình hộp đen đưa ra. Mặc dù mục tiêu của CE và các ví dụ đối kháng (adversarial examples) có một số điểm tương đồng, các thay đổi trong CE phải dễ hiểu. Ngược lại, các ví dụ đối kháng chứa nhiễu tần số cao mà mắt người không thể phân biệt được. Nhìn chung, CEs nhắm đến ba mục tiêu chính: 
- (i) tạo ra các hình ảnh gần gũi với các thay đổi thưa thớt, tức là các mẫu với thay đổi nhỏ nhất, 
- (ii) các giải thích phải thực tế và dễ hiểu đối với con người
- (iii) phương pháp tạo đối ngẫu phải tạo ra các mẫu đa dạng. 
Nói chung, CEs tìm cách làm sáng tỏ các mối tương quan đã học liên quan đến các quyết định của mô hình.

Nhiều nghiên cứu về CE sử dụng các mô hình sinh (generative models) để tạo ra những thay đổi hữu hình trong hình ảnh. Các kiến trúc này nhận biết các yếu tố để tạo ra hình ảnh gần với đa tạp hình ảnh (image manifold). Dựa trên những tiến bộ gần đây trong cộng đồng tổng hợp hình ảnh, chúng tôi đề xuất DiME: Diffusion Models for Counterfactual Explanations (Mô hình khuếch tán cho CE). DiME tận dụng các mô hình xác suất khuếch tán khử nhiễu (Denoising Diffusion Probabilistic Models - DDPM) để tạo ra các CEs. Để đơn giản, chúng tôi sẽ gọi các mô hình này là mô hình khuếch tán hoặc DDPMs. Theo hiểu biết của chúng tôi, đây là lần đầu tiên các phương pháp tổng hợp mới này được khai thác trong bối cảnh của CE.

Các mô hình khuếch tán mang lại nhiều lợi thế so với các mô hình sinh thay thế, chẳng hạn như GANs.
- Trước tiên, DDPMs có nhiều không gian tiềm ẩn; mỗi không gian kiểm soát các chi tiết thô và tinh tế. Chúng tôi tận dụng các không gian nhiễu mức thấp để tạo ra những thay đổi có ý nghĩa về mặt ngữ nghĩa trong hình ảnh đầu vào. Các không gian này chỉ mới được nghiên cứu gần đây cho tác vụ chỉnh sửa hình ảnh. 
- Thứ hai, do bản chất xác suất của chúng, chúng tạo ra tập hợp hình ảnh đa dạng. Tính ngẫu nhiên này là lý tưởng cho CE vì nhiều giải thích khác nhau có thể giải thích các lỗi của bộ phân loại. 
- Thứ ba, kết quả của Nichol và Dhariwal cho thấy DDPMs bao phủ một phạm vi rộng hơn trong phân phối hình ảnh mục tiêu. Cuối cùng, việc huấn luyện DDPMs ổn định hơn các mô hình tổng hợp tiên tiến khác, đặc biệt là GANs. Do phát triển tương đối mới, DDPMs còn nhiều khía cạnh chưa được khám phá.

Chúng tôi đóng góp một bước nhỏ vào cộng đồng XAI bằng cách nghiên cứu các không gian tiềm ẩn nhiễu mức thấp của DDPMs trong bối cảnh CE. Chúng tôi tóm tắt những đóng góp của mình như sau:

- DiME sử dụng các mô hình khuếch tán mới nhất để tạo ra các ví dụ đối ngẫu. Không giống như các mô hình sinh khác, thuật toán CE của chúng tôi không yêu cầu huấn luyện mô hình khuếch tán theo cách có điều kiện hoặc tái huấn luyện nó bằng gradient, tức là chúng tôi chỉ dựa vào một DDPM không điều kiện đã được huấn luyện.
- Chúng tôi đưa ra cách mới để tận dụng bộ phân loại (target classifier) hiện có để hướng dẫn quá trình tạo hình ảnh thay vì huấn luyện trên các mẫu nhiễu.
- Chúng tôi đạt được kết quả tiên tiến mới trên tập dữ liệu CelebA, vượt qua các nghiên cứu trước đó về CE trên các chỉ số FID, FVA và MNAC cho thuộc tính "Smile" và chỉ số FID, MNAC cho thuộc tính "Young."
Chúng tôi chỉ ra rằng MNAC mang lại cảm giác sai lầm trong việc đánh giá đúng các đối ngẫu và đề xuất một thước đo mới gọi là Correlation Difference để đánh giá các mối tương quan giả một cách tinh tế trong bối cảnh CE.

## 2. Related work
### 2.1 Phân loại XAI:
Công trình của chúng tôi đóng góp vào lĩnh vực Trí tuệ nhân tạo có thể giải thích được (XAI), trong đó có thể phân biệt hai nhóm phương pháp chính: phương pháp giải thích ngay từ thiết kế (interpretable-by-design) và phương pháp giải thích hậu kỳ (post-hoc approaches).

- Nhóm thứ nhất (interpretable-by-design) bao gồm các cơ chế có thể giải thích được được tích hợp ngay trong giai đoạn thiết kế của mô hình [2, 3, 6, 9, 22, 40, 69].
    - Learning decision trees recurrently through communication.
        - Mục tiêu: Kết hợp sức mạnh của cây quyết định (decision trees) và mạng nơ-ron hồi tiếp (recurrent neural networks) để đạt được tính giải thích và hiệu quả cao trong học máy.
        - Phương pháp:
            - Xây dựng một cơ chế học quyết định lặp lại (recurrent decision-making) thông qua các nút cây quyết định.
            - Mô hình cho phép các nút giao tiếp với nhau, tận dụng thông tin từ mạng hồi tiếp để cập nhật và tinh chỉnh quá trình phân loại.
    - Towards robust interpretability with selfexplaining neural networks.
        - Mục tiêu: Tạo ra các mô hình học sâu tự giải thích (self-explaining), đảm bảo tính minh bạch và khả năng giải thích mà không làm mất đi hiệu năng dự đoán.
        - Phương pháp:
            - Kết hợp các giải thích tuyến tính cục bộ (local linear explanations) vào cấu trúc mô hình.
            - Thiết kế mô hình sao cho đầu ra có thể được liên kết trực tiếp với các đặc trưng quan trọng, tăng cường tính minh bạch và độ bền vững.
    - *Convolutional dynamic alignment networks for interpretable classifications.*
        - Mục tiêu: Phát triển mạng nơ-ron tích chập (CNN) có khả năng cung cấp các giải thích linh hoạt và dễ hiểu trong bài toán phân loại.
        - Phương pháp:
            - Tích hợp căn chỉnh động (dynamic alignment) vào CNN, làm nổi bật các vùng ảnh hoặc đặc trưng quan trọng nhất cho dự đoán.
            - Sử dụng một cơ chế tập trung (attention mechanism) để điều chỉnh cách các đặc trưng được liên kết với kết quả.
    - *This looks like that: Deep learning for interpretable image recognition*
        - Mục tiêu: Cung cấp giải thích dễ hiểu về các dự đoán trong bài toán nhận diện ảnh bằng cách so sánh với các mẫu đã biết.
        - Phương pháp:
            - Sử dụng các mẫu nguyên mẫu (prototypes), đại diện cho các đặc điểm đặc trưng.
            - Khi dự đoán, mô hình so sánh các vùng trong ảnh đầu vào với các nguyên mẫu để giải thích dự đoán bằng cụm từ "This looks like that."
    - Interpretable and accurate fine-grained recognition via region grouping.
    - Neural prototype trees for interpretable finegrained image recognition
        - Mục tiêu: Kết hợp mô hình cây quyết định (decision trees) và mạng nơ-ron để giải quyết bài toán nhận diện chi tiết với giải thích trực quan.
        - Phương pháp:
            - Sử dụng các nguyên mẫu (prototypes) làm nút trong cây quyết định.
            - Mỗi nhánh của cây liên kết với các đặc trưng cụ thể, cho phép mô hình đưa ra dự đoán dựa trên các quyết định theo cấp bậc.
    - Interpretable convolutional neural networks
        - Mục tiêu: Phát triển một kiến trúc CNN có khả năng giải thích các quyết định dự đoán mà không làm giảm hiệu suất.
        - Phương pháp:
            - Thiết kế các lớp giải thích (explanation layers), làm nổi bật các đặc trưng quan trọng nhất trong đầu vào.
            - Áp dụng các ràng buộc lên trọng số và cấu trúc của mạng để đảm bảo tính giải thích mà không làm mất thông tin.


- Nhóm thứ hai (post-hoc) nhắm tới việc hiểu hành vi của các mô hình học máy hiện có mà không cần thay đổi cấu trúc nội tại của chúng. Phương pháp của chúng tôi thuộc nhóm thứ hai này.

Hai nhóm phương pháp này có các mục tiêu và lợi thế khác nhau:
- Một lợi ích của các phương pháp hậu kỳ là chúng có thể dựa vào các mô hình hiện tại vốn đã được chứng minh có hiệu năng tốt.
- Ngược lại, các phương pháp giải thích ngay từ thiết kế thường dẫn đến sự đánh đổi về hiệu năng (performance trade-off).

### 2.2 Phương pháp hậu kỳ (post-hoc):
- Mục tiêu của các phương pháp hậu kỳ là tìm hiểu hành vi của các mô hình học máy (ML) hiện có mà không cần phải thay đổi cấu trúc nội bộ của chúng.
- Phương pháp của chúng tôi thuộc nhóm này. Hai nhóm này có các mục tiêu và ưu điểm khác nhau; một lợi ích của phương pháp hậu kỳ là chúng dựa trên các mô hình hiện có vốn đã có hiệu suất tốt, trong khi các phương pháp giải thích từ khi thiết kế thường dẫn đến việc đánh đổi hiệu suất.
- Trong lĩnh vực các phương pháp hậu kỳ, có một số hướng nghiên cứu đã được khám phá:
    - Chiến lược chưng cất mô hình (Model Distillation): Các chiến lược này tiếp cận khả năng giải thích bằng cách khớp một mô hình có thể giải thích được với các dự đoán của mô hình hộp đen [13, 58].
        -  A peek into the reasoning of neural networks: Interpreting with structural visual concepts.
        -  Learning global additive explanations for neural nets using model distillation

    - Giải thích dưới dạng văn bản:
        - **Một số phương pháp tạo ra các giải thích dưới dạng văn bản [17, 43, 66].**
            - Generating Visual Explanations
                - Mục tiêu:
                    - Cung cấp giải thích trực quan cho các dự đoán của mô hình học sâu, giúp người dùng không chỉ hiểu tại sao mô hình đưa ra quyết định, mà còn nhìn thấy bằng chứng trực quan liên quan đến quyết định đó.
                - Phương pháp:
                    - Sử dụng mạng nơ-ron tích chập (CNNs) kết hợp với cơ chế tập trung (attention mechanism) để xác định các khu vực quan trọng trong hình ảnh.
                    - Tạo ra các saliency maps hoặc vùng nổi bật giúp chỉ ra những phần ảnh mà mô hình tập trung khi đưa ra quyết định.
                    - Tích hợp các lời giải thích bằng ngôn ngữ tự nhiên (nếu có) để nâng cao sự trực quan hóa.

            - Multimodal Explanations: Justifying Decisions and Pointing to the Evidence
                - Mục tiêu:
                    - Cung cấp các giải thích đa phương thức (multimodal explanations) kết hợp giữa văn bản và hình ảnh, vừa giải thích quyết định của mô hình vừa chỉ ra bằng chứng hỗ trợ từ dữ liệu đầu vào.
                    - Tăng cường tính minh bạch trong các hệ thống AI hoạt động với dữ liệu đa phương thức (ví dụ: văn bản, hình ảnh, video).
                - Phương pháp:
                    - Kết hợp các mô hình học sâu xử lý ngôn ngữ tự nhiên (NLP) và thị giác máy tính (computer vision) để tạo ra các giải thích đồng thời.
                    - Cơ chế đồng nhất hóa (alignment mechanism): Đồng bộ hóa các đặc trưng giữa các modalities (ví dụ: lời giải thích văn bản liên kết với vùng ảnh cụ thể).
                    - Tích hợp attention mechanism để tập trung vào các vùng ảnh và đoạn văn bản quan trọng, phù hợp với kết quả dự đoán.

            - f-VAEGAN-D2: A Feature Generating Framework for Any-Shot Learning
                - ?

    - Giải thích thông tin hình ảnh:
        - Đối với việc giải thích thông tin hình ảnh, tầm quan trọng của đặc trưng (feature importance) là cách tiếp cận phổ biến nhất, thường được thực hiện dưới dạng các bản đồ trọng số (saliency maps). 
        - Các bản đồ này được tính toán bằng cách: 
            - sử dụng gradient trong mạng [8, 27, 33, 53, 63, 72].
                - Gradcam++: Generalized gradient-based visual explanations for deep convolutional networks
                - Cameras: Enhanced resolution and sanity preserving class activation mapping for image saliency.
                - Relevance-cam: Your model already knows where to look.
                - Gradcam: Visual explanations from deep networks via gradient-based localization.
                - Score-cam: Score-weighted visual explanations for convolutional neural networks.
                - Learning deep features for discriminative localization. 
                - (Này chắc thiên về test code hơn)
            - bằng cách tạo ra nhiễu trên hình ảnh [45, 46, 61, 68].   
                - RISE: randomized input sampling for explanation of black-box models.
                - Black-box explanation of object detectors via saliency maps.
                - Iterative and adaptive sampling with spatial attention for blackbox model explanations.
                - Visualizing color-wise saliency of black-box image classification models.
                - (Này chắc thiên về test code hơn)
                
    - Phương pháp phân bổ khái niệm (Concept Attribution):
        - Các phương pháp này tìm kiếm các đặc điểm phổ biến nhất mô tả một lớp hoặc một mẫu cụ thể. Các thuật toán này:
            - trực quan hóa các khái niệm có thể hiểu được bởi con người, chẳng hạn như kết cấu hoặc hình dạng.
                - Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV)
                    - ?
            - tìm kiếm các khái niệm có thể hiểu được bởi con người, chẳng hạn như kết cấu hoặc hình dạng. (chia nhỏ hoặc lấy các điểm - point - trong ảnh để xác định class)
                - Interpreting with structural visual concepts.
                    - Mục tiêu:
                        - Giải thích các dự đoán của mô hình học sâu bằng cách sử dụng các khái niệm cấu trúc trực quan (structural visual concepts), như các bộ phận của đối tượng hoặc cấu trúc hình học.
                        - Cung cấp giải thích dựa trên các đặc điểm mà con người có thể hiểu và liên kết với dữ liệu đầu vào.
                    - Phương pháp:
                        - Trích xuất các concepts (khái niệm) từ dữ liệu đầu vào thông qua các lớp biểu diễn trong mô hình.
                        - Ánh xạ các khái niệm này tới các đặc trưng cấu trúc của đối tượng, ví dụ: mắt, mũi của một khuôn mặt hoặc cạnh và góc của hình học.
                        - Sử dụng các khái niệm này để xây dựng lời giải thích trực quan và dễ hiểu hơn.

                        ![alt text](<../Images/Interpreting with structural visual concepts..png>)
                - Towards automatic concept-based explanations.
                    - ?
                    - Mục tiêu:
                        - Tự động hóa quá trình sinh các giải thích dựa trên khái niệm (concept-based explanations) mà không cần sự can thiệp thủ công.
                        - Cải thiện tính hiệu quả và khả năng tái sử dụng của các khái niệm trong giải thích mô hình học sâu.
                    - Phương pháp:
                        - Tạo ra các khái niệm tự động bằng cách:
                        - Sử dụng biểu diễn không gian tiềm ẩn của mô hình để nhóm các đặc trưng tương tự.
                        - Ánh xạ các nhóm đặc trưng này tới các concepts có ý nghĩa.
                        - Đo lường mức độ ảnh hưởng của từng khái niệm đến dự đoán thông qua các phương pháp như importance scoring hoặc concept activation.
                - On completeness-aware concept-based explanations in deep neural networks.
                    - ?
                - Interpretable Basis Decomposition for Visual Explanation.
                    - ...
                    - Mục tiêu:
                        - Đảm bảo rằng các giải thích dựa trên khái niệm (concept-based explanations) là đầy đủ (complete), nghĩa là chúng bao phủ đầy đủ các yếu tố ảnh hưởng đến quyết định của mô hình.
                        - Đánh giá mức độ hoàn thiện của các khái niệm được sử dụng trong giải thích.
                    - Phương pháp:
                        - Đề xuất một khung phân tích completeness-aware:
                            - Định nghĩa tính đầy đủ của các giải thích dựa trên khái niệm bằng cách đo lường lượng thông tin mà các khái niệm bao hàm trong quyết định.
                            - Sử dụng các phương pháp như Shapley values để đo lường mức độ đóng góp của từng khái niệm.
                        - Tích hợp cơ chế đánh giá tính đầy đủ để đảm bảo giải thích không bỏ sót các yếu tố quan trọng.

### 2.3 Counterfactual Explanations - CE
- Tầm quan trọng:
    - CE là một nhánh của các phương pháp giải thích hậu kỳ. Chúng đặc biệt hữu ích trong việc biện minh cho các quyết định được đưa ra tự động bởi các thuật toán [62].
        -  Counterfactual Explanations Without Opening the Black Box: Automated Decisions and the GDPR. (bài tổng hợp)
    - Về cơ bản, một CE là sự thay đổi nhỏ nhất nhưng có ý nghĩa đối với một mẫu đầu vào để đạt được kết quả mong muốn của thuật toán.

- Các phương pháp hiện tại:
    - Một số phương pháp gần đây [15, 64] khai thác các vùng trong ảnh truy vấn và một bức ảnh khác được phân loại khác để thay đổi sự xuất hiện ngữ nghĩa, tạo ra các ví dụ đối ngẫu.
        - Counterfactual Visual Explanations
        - IMAGINE: Image Synthesis by Image-Guided Model Inversion
    
    - Các nghiên cứu khác [52, 62] tận dụng gradient của ảnh đầu vào với nhãn mục tiêu để tạo ra các thay đổi có ý nghĩa.
        - Generating Interpretable Counterfactual Explanations By Implicit Minimisation of Epistemic and Aleatoric Uncertainties
        - Counterfactual Explanations Without Opening the Black Box: Automated Decisions and the GDPR
    
    - Ngược lại, [1] tìm kiếm các mẫu nguyên mẫu mà ảnh phải chứa để thay đổi dự đoán của nó. Tương tự, [36, 47] theo một thuật toán dựa trên nguyên mẫu để tạo ra các giải thích.
        - Cocox: Generating conceptual and counterfactual explanations via fault-lines
        - Interpretable Counterfactual Explanations Guided by Prototypes
        - Face: Feasible and actionable counterfactual explanations. 
    
    -  "Deep Image Priors" [59] và "Invertible CNNs" [23] đã chứng minh khả năng tạo ra các ví dụ đối ngẫu.
        -  Designing counterfactual generators using deep model inversion.
        -  Ecinn: efficient counterfactuals from invertible neural networks
    
    - Các phân tích lý thuyết [25] đã tìm thấy những điểm tương đồng giữa các CE và các cuộc tấn công đối kháng (adversarial attacks).
        - On Relating Explanations and Adversarial Examples
            - Nghiên cứu này tìm hiểu mối quan hệ giữa giải thích mô hình (explanations) và ví dụ đối kháng (adversarial examples), nhằm:
                - Làm rõ cách các đặc trưng quan trọng (theo giải thích) bị khai thác để tạo ra ví dụ đối kháng.
                - Cải thiện cả tính minh bạch (explainability) và độ bền vững (robustness) của mô hình trước các tấn công đối kháng.
            - Phương pháp:
                - Phân tích mối liên hệ:
                    - Nghiên cứu sự trùng lặp giữa các đặc trưng quan trọng trong giải thích (saliency maps) và các vùng dễ bị tổn thương (sensitive regions) của mô hình.
                - Khai thác Adversarial Examples:
                    - Tạo ví dụ đối kháng dựa trên các đặc trưng quan trọng từ các giải thích, để kiểm tra độ bền vững của mô hình.
                - Tăng cường mô hình:
                    - Đề xuất phương pháp kết hợp giữa giải thích và đào tạo đối kháng (adversarial training) để:
                        - Làm rõ hơn các đặc trưng quan trọng.
                        - Tăng khả năng chống lại tấn công đối kháng.

- Vai trò của kỹ thuật generation:
    - Kỹ thuật sinh là yếu tố then chốt để tạo ra dữ liệu gần với đa tạp hình ảnh (image manifold). Ví dụ, [12] tối ưu hóa phần dư của hình ảnh trực tiếp bằng cách sử dụng một bộ mã hóa tự động (autoencoder) làm bộ điều chuẩn.
        - Explanations based on the Missing: Towards Contrastive Explanations with Pertinent Negatives
            - ?
            - Giải thích tương phản: Thay vì chỉ giải thích tại sao mô hình dự đoán một kết quả cụ thể, nghiên cứu này tập trung vào việc giải thích tại sao mô hình không dự đoán một kết quả khác.
    - Các nghiên cứu khác đề xuất sử dụng mạng sinh để tạo CE, bao gồm các mạng không điều kiện [41, 48, 54, 71] hoặc có điều kiện [34, 55, 60].
        - Không điều kiện:
            - Countergan: Generating realistic counterfactuals with residual generative adversarial nets
                - Nghiên cứu về việc sử dụng mạng đối kháng sinh (GAN) để tạo ra các counterfactuals
            - Beyond Trivial Counterfactual Explanations with Diverse Valuable Explanations
                - Sử dụng các kỹ thuật tối ưu hóa để tạo ra nhiều giải thích phản thực trong một không gian giải thích rộng lớn.
            - GANMEX: One-vs-one Attributions Using GAN-based Model Explainability
                - Sử dụng Generative Adversarial Networks (GANs) để tạo ra các mẫu đại diện và kiểm tra sự khác biệt giữa các lớp.
            - Generating Natural Adversarial Examples 
                - Tập trung vào việc tối ưu hóa để:
                    - Giảm thiểu sự khác biệt giữa ví dụ đối kháng và dữ liệu gốc.
                    - Đảm bảo mẫu đối kháng vẫn giữ được tính thực tế (naturalistic constraints).
        - Có điều kiện:   
            - Generative counterfactual introspection for explainable deep learning.
                - Sử dụng các mô hình sinh như GAN (Generative Adversarial Networks) hoặc VAE (Variational Autoencoders) để tạo ra các phiên bản thay thế của dữ liệu đầu vào (counterfactuals).
            - Explanation by Progressive Exaggeration:
                - Phóng đại dần dần (progressive exaggeration): Tăng cường các đặc trưng quan trọng qua nhiều bước nhỏ để tạo ra một chuỗi các trạng thái trung gian.
            -  Conditional generative models for counterfactual explanations. 
                - Sử dụng Conditional Generative Models, như Conditional Variational Autoencoders (CVAE) hoặc Conditional Generative Adversarial Networks (CGAN), để sinh các dữ liệu phản thực.
                - Các bước chính:
                    - Điều kiện hóa (Conditioning): Đưa ra mục tiêu dự đoán mới (ví dụ: thay đổi kết quả từ "phủ định" sang "khẳng định").
                    - Sinh dữ liệu phản thực: Tạo ra dữ liệu đầu vào mới, thay đổi các yếu tố phù hợp để đạt được kết quả mong muốn.
                    - Đảm bảo tính tự nhiên: Đặt các ràng buộc để dữ liệu phản thực vẫn nằm trong không gian dữ liệu hợp lệ và giữ nguyên các đặc điểm không thay đổi.

    
### 2.4 Diffusion Models:
- Ứng dụng:
    Các mô hình khuếch tán gần đây đã trở nên phổ biến trong lĩnh vực nghiên cứu tổng hợp hình ảnh [19, 56]. Ví dụ, DDPM đã được áp dụng cho các tác vụ như inpainting [49], tổng hợp hình ảnh có điều kiện và không điều kiện [10, 19, 42], siêu phân giải (super-resolution) [50], và thậm chí các nhiệm vụ cơ bản như phân đoạn (segmentation) [5].
    - Tổng hợp hình ảnh:
        - DDPM
        - Denoising diffusion implicit models.
    - Inpainting:
        - Palette: Image-to-image diffusion models
    - Tổng hợp có điều kiện và ko điều kiện:
        - Conditioning method for denoising diffusion probabilistic models.
            -> đổi phong cách ảnh
        - Improving DDPM
        - DDPM
    - Super-resolution:
        - Image superresolution via iterative refinement.
    - Segmentation:
        - Labelefficient semantic segmentation with diffusion models.
- Ưu điểm:
    Nhiều nghiên cứu [20, 57] đã chỉ ra rằng cách tiếp cận dựa trên điểm (score-based) và khuếch tán là các cách tiếp cận thay thế để khử nhiễu (denoising) trong quá trình lấy mẫu ngược nhằm sinh dữ liệu.
    - A variational perspective on diffusion-based generative models and score matching
        - ?
    - Scorebased generative modeling through stochastic differential equations
        - ?

- Hạn chế:
    Do quá trình tạo ra hình ảnh theo từng bước (recursive generation process), việc lấy mẫu từ DDPM tốn nhiều thời gian. Nhiều công trình đã nghiên cứu các cách tiếp cận thay thế để tăng tốc quá trình tạo hình ảnh [31, 65].
    -  On fast sampling of diffusion probabilistic models
    -  Learning to efficiently sample from diffusion probabilistic models. 
- Điểm khác biệt của nghiên cứu này:
    Phương pháp gần đây của [11] tập trung vào việc tạo hình ảnh có điều kiện với các mô hình khuếch tán bằng cách huấn luyện một bộ phân loại cụ thể trên các mẫu nhiễu để thiên hướng quá trình sinh hình ảnh.
    - Công trình của chúng tôi (nhóm tác giả) có một số điểm tương đồng với phương pháp này, nhưng trong trường hợp của chúng tôi, việc giải thích một bộ phân loại hiện có được huấn luyện duy nhất trên các mẫu sạch (clean instances) đặt ra một thách thức bổ sung.
    - Hơn nữa, không giống như các phương pháp khuếch tán trước đây, chúng tôi thực hiện quá trình chỉnh sửa hình ảnh từ một bước trung gian thay vì bước cuối cùng.

## 3. Methodology

**github: https://github.com/guillaumejs2403/DiME**

- Tóm tắt quy trình:
    - Dataset: CelebA gồm các mẫu, 1 mẫu gồm 1 ảnh và 40 label (Nguồn: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
    - Train diffusion model cho các mẫu và nhãn tương ứng của mẫu.
    - Train classifier model (sử dụng dense net) tạo output cho 40 thuộc tính.
    - Inference:
        - Trong quá trình gỡ noise sử dụng guided-diffusion để Loss nhỏ nhất.
        - Loss  bao gồm Lclass và Lperc:
            - Lclass là loss của ảnh target với class được chỉ định (nhãn được đảo ngược).
            - Lperc là perceptive loss với input là src image và  target image sử dụng model VGG19.
                - Cơ bản về perceptive loss: Thông qua các layer của VGG19 cho ra feature của src image và targe image. (Theo như tìm hiểu là nó sẽ thiên về ngữ nghĩa).
                - Sử dụng MSE loss.
        - Lclass có chức năng kiểm soát class output đầu ra từ diffusion model.
        - Lperc có chức năng kiểm tra hình ảnh đầu ra giống với hình ảnh gốc.
- Ưu điểm:
    - Chỉ cần train 2 lần gồm 1 lần train Diffusion model và 1 lần train classifier (DenseNet).
    - Giữ được bản chất CE (tối ưu Lclass và Lperc).
    - Đa dạng

## 4. Experiment
Các metrics:
- CD (Correlation Difference):
    - Tính corrs: corrs là tương quan giữa nhãn "Smiling" và các thuộc tính khác trong tập dữ liệu.
    - Tính result: Tính sự khác biệt giữa ảnh gốc và ảnh đối chứng.
    - So sánh result và corrs -> so sánh giữa tương quan dự đoán và tương quan thực tế?
- FVA (Feature Vector Agreement): Đầu vào là src img và target img
    - Đưa qua Resnet50_128 để lấy Feature Vector 
    - Tính FVA = 1 - cosim
- LPIPS (Learned Perceptual Image Patch Similarity): đánh giá mức độ tương đồng giữa hai hình ảnh dựa trên nhận thức của con người.
    - Đưa ảnh x và ảnh y qua NN (cụ thể là VGG) 
    - Tính khoảng cách ở mỗi layer
    - Trọng số hoá khoảng cách
- MNAC (Misclassification by Nearest-Adversarial Class):
    - Classifier ở đây là Resnet50_128
    - MNAC tính số lượng nhãn khác nhau khi so sánh dự đoán (pi) của ảnh gốc với dự đoán (pc) của ảnh đối chứng từ ResNet.
