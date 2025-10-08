Research notes, brainstorming, thoughts, and everything:
Document structure:
1-	Firstly, and most importantly, I have sensei’s notes, comments, and suggestion, most important
2-	Secondly, I have some senpais comments
3-	Thirdly, I have some interesting points I thought of
4-	Fourthly, I have some scattered thoughts that were originally originally by an AI model, don’t assign the highest priority to it.
5-	Fifth, I wrote multiple iterations of the research proposal
Sensei’s notes, comments, and suggestions (and other things) (most important):
“I think it would be better if you could increase the number of bands by extracting from the source. How about conducting an analysis using DS as an initial trial? Did you read my TPAMI2015 paper? You can find similar examples in the paper. This is the paper link https://ieeexplore.ieee.org/document/7053916 “
“I am interested in how the DS work to extract the difference components between before and after satellite subspaces. I know well that the approach using a linear DS is still native and straightforward. But, I believe that it is a good starting point for future deep research.”
“Hi Abdelrahman,I wondered if we could apply the temporal analysis using subspace representation  to your research proposal.  I expect that the fist and second difference subspaces work on the temporal analysis of satellite mage when we could a satellite image with multiple band channels as a subspace. How about searching some methods using subspace representation on satellite image analysis?
I recommend you read my ArXiv paper that proposes the second DS as well.If you could find an open dataset, we can start studying your idea soon.If you have any questions on subspace representation, including linear algebra, you can ask Pedro and Santos about them.Please let me know when you find a valid dataset of satellite images.”

“Dear Abdo,  do you also like theoretical research using deep math? Kazu”

“there are many many types of change detection methods. You can find several survey papers on the internet. How about reading such papers to catch the overview, including the necessary math? Besides, I recommend our paper using first and second DSs as one possible approach, in which the concept of DSs is applied to a task of change detection.
https://www.arxiv.org/pdf/2409.08563 The following paper contains several mistakes, but the reference list can be helpful for you.
https://arxiv.org/abs/2303.17802”

“Do you want to read a text book of SSA  (Singular spectrum analysis)? You may be able to find it in the lab. Suto-kun has already extended the concept of SSA using vectors to that using subspaces by introducing DS.”

“For the methods using SSA (Singular spectrum analysis), you can find several papers from my publication list. For example, Lincon, Bernadro and Maha also used SSA in their works.Besides, the concept of slow feature analysis also is useful.
Kobayashi-san at AIST & Prof. UT proposed the following idea and then Suzana applied it to her PhD work.
https://staff.aist.go.jp/takumi.kobayashi/publication/2017/BMVC2017.pdf
https://www.sciencedirect.com/science/article/pii/S2666827023000464”
Senpais’ comments:
First senpai:
•	like why are we using the land-use and damage assessments? Why are we combining them? What do we really wanna achieve? Like for example helping with reconstruction efforts? What does that really mean? Like for example do you wanna look a place before and after it was damaged, and decide according to the damage that happened to it after a disaster what is the optimal way to build it so that it doesn’t get as dangerously destroyed before or to limit damage or to understand how to keep the infrastructure of the building more stable? And how would you do that exactly? What is really the benefit of all of this?. No, but really, what is the purpose of this research, what is it really you wanna achieve?
•	it's like, why are you using these tasks? How would you really utilize them? Say you got the data from satellites to a certain place that got affected by a disaster, you got the data, did dimensionality reduction with SSC, did damage assessment to a certain area with U-Net, and land-use also with U-Net. What then? What is it you wanna do afterwards? What is it really proposed here? Its like there’s this missing piece. Its fine that you want to say disaster resilience, but what is it really you wanna do? Make things more clear.
•	The task isn’t understandable from my perspective, is it to segment areas that are damaged, for example, from an input satellite image, is the task to return binary images where white color means damaged area, black is not damaged? Or classify the damage of an area? From the title, I think it's a little bit confusing. I think focusing on one part only is better, (this is me speaking, the focus is assessing damage in levels, like is the building 75% damaged? 56% damaged? Fully damaged? Partially damaged? So I think it's basically classifying the damage of an area. How would we work to clear the confusion my senpai mentioned)
•	Second, I don't really get the idea of using sparse subspace clustering. Is this what sensei was proposing back then? And also, the second paragraph doesn't have a solution yet in this proposal. I thought that you wanted to propose something to address the limitation of deep learning methods.
•	how to combine the land-use and damage assessment tasks? What is the method or methodology I would use for that? It's not clear. how to make sense of the combining of these tasks in a concrete way, not some vague general sentences that don’t explain the method or the why and the how properly.
•	Do you think you wanna use LLMs? Maybe that's good, but I don’t know for sure.

Second senpai (Gulpi-san):
•	Clearly define the task, specifying whether it involves segmenting damaged areas (binary output) or classifying damage levels (e.g., percentage or categories).
•	Articulate the objective and purpose of the research, explaining why land-use and damage assessments are combined and how the results will be utilized.
•	Narrow the scope for better clarity and focus, potentially prioritizing damage classification by levels.
•	Justify the use of Sparse Subspace Clustering (SSC) by explaining its role and contribution to addressing the limitations of deep learning, and clarify if it stems from a previous suggestion or offers a novel solution.
•	Provide a detailed methodology for integrating land-use and damage assessment tasks, avoiding vague or general statements.
•	Address the gap in the proposal by offering a specific solution to the computational inefficiency of deep learning models.
•	Explain the post-assessment application, detailing how the outputs from satellite/UAV data, SSC, and U-Net will be used for actionable outcomes, such as reconstruction or disaster mitigation.
•	Consider the inclusion of Large Language Models (LLMs) as a potential enhancement, though this requires further justification and alignment with the framework.

Interesting ideas:
•	dictionary learning to extract features.
•	Fermi data, sound from topography? Destruction? Building?
•	NEW GOAL (Potential main?): understand how damage evolves over time (temporal dynamics), for that we can use DMD (dynamic mode decomposition)
•	LASSO: Standard method of sparse modeling
•	Fourier Transform, Fourier analysi, fourier on time series satellite data.

Points about research:
•	NEW IDEA: combine LiDAR data, SAR and InSAR data, and other sensors to infer knowledge?
•	captures structural patterns in large datasets. 
•	Clearly define the task, specifying whether it involves segmenting damaged areas (binary output) or classifying damage levels (e.g., percentage or categories).
•	Articulate the objective and purpose of the research, explaining why land-use and damage assessments are combined and how the results will be utilized.
•	Task in not clearly defined, what do I wanna do? There’s a missing piece
•	Combining Land use and damage assessment, Methods to combine between both maybe..?
•	lightweight deployment to edge devices? on-the-ground application in post-disaster scenarios and urban planning efforts  (UAVs?)
•	Explain the post-assessment application, detailing how the outputs from satellite/UAV data, SSC, and U-Net will be used for actionable outcomes, such as reconstruction or disaster mitigation.
•	More detailed methodology? Maybe?
•	Offer a specific solution to the computational inefficiency of deep learning models, subspace methods on UAV?
•	How will the outputs from satellite/UAV data, SSC, and U-Net be used for actionable outcomes? More details are needed, vagueness makes no sense
•	The Damage Assessment information and the Land-use information, will it be a heatmap? Grid numbers? How will we present that information?
•	Will we use SSC to compress data?
•	Explain more what xBD does, xBD used to Damage assessment
•	“Sell your fish”, maybe deployment on UAV as the main focus? Maybe running on drones is a good reason
•	SSC -> drones, U-Net -> computer
•	Current post-disaster assessment models often rely on Deep Learning methods, what do these models do, for example identifying..
•	How do we give insights from two heatmaps? Will Insights = heatmap Output -> ?
•	This is a mixed interfeild
•	focus on Drones images UAV maybe? (senpai advised me this)
•	Processing images in drone (resource constraints)
•	Maps for multiple infrastructures (schools, hospitals, etc..)? a way to construct the maps, A method to build maps?
•	Maybe we can use Multi-criteria decision analysis (MCDA), Multi criteria analysis, MCDM?
•	Built a heatmap for MCA, made subspaces for each year 2011, 2012, 2013 for example, compare subspaces and see similarity
•	Multi criteria analysis using satellite images for Disaster Resillience
•	“processing satellite imaging remote sensing computer vision to determine civil engineering infrastructure route”?
•	Try to create from satellite images a perspective of a city and ask a generative model to create a perspective of the city depending on where you are and where you are looking.
•	Using subspaces and remote sensing to determine the optimum placement for urgent infrastructure(hospitals, evacuation centers)
•	processing satellite imaging remote sensing computer vision to determine civil engineering infrastructure route.
•	An AI model that processes satellites images for a certain region, it looks at its contours, terrain, climate, and other general factors, in order to determine what is the best way to build such a region, as in, Urban and architectural planning. Homes, hospitals, other facilities, determine where and how and what.
•	Do you remember the news about the lab that built the mushroom thing that detects the best route to train metro-lines in Tokyo, maybe we can get inspiration form that? see how the ants build their homes/colonies, make OpenCV datasets from that, maybe it could be useful, apply the research idea on ant colonies and extract wisdom.
•	Also, hospitals, utilizing best routes for evacuation of patients, transporting and stuff neighborhood
•	Damage analysis information as heatmaps?
•	Damage analysis -> heatmaps
•	LU -> grid numbers
•	SSC to compress data
•	Xbd explain more, we use xbd for damage analysis
•	Sell my fish, deployment on UAV could be a unique unexlored region 
•	Running on drones is a good reason
•	SSC on drone, U-Net on computers
•	Current post-disaster assessment models often rely on Deep Learning methods, what do these models, for example identifying
•	How do we give insights from two heatmaps?
•	Insights = heatmap Output -> ?
•	Multi criteria analysis
•	Multi criteria decision analysis
•	mix interfield
•	MCDM
•	Multi criteria analysis using satellite images for Disaster resillience
•	Maybe extract wisdom from the Multi criteria analysis using satellite images for GIS Saudi Arania groundwater mapping paper?
•	Using subspace methods to calculate similarly 
•	Zero shot learning (could be a great idea, important)
•	You could focus on Drones images UAV for the research, could be a unique idea.
•	Processing images in drone (resource constraints)
•	Landslide detection
•	Automation of detection of landslides
•	Landslide satellite images
•	Maps for multiple infrastcures (schools, hospitals, etc..)? a way to construct the maps, 
•	A method to build maps 
•	Maybe we could built destruction maps using drones.
•	Built a heatmap for MCA, made subspaces for each year 2011, 2012, 2013 for example, compare subspaces and see similarity.
•	What is the task really about? What is the objective? What is accomplished or the thing to be accomplished? And specify it, not some general sentences that are vague.
•	Satellite image disaster resilience evacuation routes?
•	Delta encoder, could be useful, sounds important
•	Critical infrastructure analysis assessment
•	Tracking refugees movement
•	Depth, lidar data 3d info, shape of change and angle of change, visual changes, shapes and shifts in space
•	SAR data cloud, hidden data
•	Combine all data
•	Idea human mobile phone on-presence using pictures or videos to capture building structural damage.  
•	There’s a missing piece of combining Land use and damage assessment
•	Methods to combine.
•	processing satellite imaging remote sensing computer vision to determine civil engineering infrastructure route
•	Remote sensing survey infrastructure 
•	Using subspaces and remote sensing to determine the optimum placement for urgent infrastructure (hospitals, evacuation centers)
•	An AI model that processes satellites images for a certain region, it looks at this contours, climate, topography, terrain, enviroment, and other general factors, to determine what is the best way to build such region, as in civil/urban/infrastructure engineering. Homes, hospitals, other facilities, determine where and how and what.
•	processing satellite imaging remote sensing computer vision to determine civil engineering infrastructure route, The lab that built the mushroom thing that detects best route to train metro-lines in Tokyo, utalize that, see how the ants build their homes/colones, make OpenCV datasets from that, maybe it could be useful, apply the research idea on ant colonies and extract wisdom.
•	Also, hospitals, utilizing best routed for evacuation or patients transporting and stuff neighborhood 
•	increased frequency of natural and man-made disasters requires the need for post-event damage assessment and land-use analysis for disaster resilience purposes, including the reconstruction phase
•	urban/disaster resilience: refers to the ability of cities to effectively analyze, respond and recover from disasters, hence that efficient and fast recovery requires rapid and precise insights into land-use patterns and structural damage 
•	Focus is land-use and damage classification
•	Aiming to offer insights for urban planners, deployable tools to improve disaster resilience, scalable tools for urban planners in both developed and developing regions, to enhance urban planning and resilience against recurring disasters
•	Japan, disaster-prone place for earthquakes and tsunamis, similarly, war-torn regions, where infrastructure is decimated, demeaning scalable tools to assist in reconstruction
•	DL techniques, while accurate, are computationally expensive and less practical in resource-constrained environments
•	Why combine the independent tasks of damage assessment and land-use? Damage assessment identifies the impact on infrastructure, while land-use analysis helps understand the spatial distribution of resources and human activity. Together, they provide a comprehensive view of the affected areas, enabling more targeted and efficient recovery strategies, optimal resource allocation, and better urban resilience.
•	a hybrid framework combining subspace and deep learning methods
•	UAV-acquired imagery
•	Datasets such as: Sentinel-2, EuroSAT, xView2, and xBD (most important), publicly accessible, annotated datasets are important
•	Sentinel-2 first for training to classify land-use features (e.g., urban, rural, vegetation) across high-resolution images, xBD secondly, since it has high-resolution satellite imagery specifically annotated for pre- and post-disaster damage assessment, we use part of it for a second round of training, and the other part of testing and validation, then validate and test on combination of both and other datasets
•	For datasets, perform augmentation, applying transformations like rotation, flipping, scaling, or brightness adjustment to input images like destroyed building. It helps the model generalize better and accurately describe the damage percentage.
•	We’ll use Sparse Subspace Clustering (SSC), it effectively reduces the dimensionality of high-resolution satellite data while preserving important features, captures structural patterns in large datasets 
•	CNNs (NO, U-Nets? SenNet? Hmmmm, seems like no) for extracting spatial features, SM complementing deep learning models by improving computational efficiency and interpretability 
•	captures structural patterns in large datasets. 
•	Compare against baseline models (e.g., standalone CNNs, traditional classifiers) on benchmark datasets
•	lightweight deployment to edge devices (UAVs?) using PyTorch
•	F1-score and IoU as metrics against other approaches
•	on-the-ground application in post-disaster scenarios and urban planning efforts (UAVs)?
•	Enhance the framework with real-time data from drones or IoT sensors for dynamic infrastructure planning
•	Expand collaborations to include government agencies, NGOs, and international research institutions to scale the impact of the research.
•	Adapt the methodology for use in other domains, such as climate change monitoring, deforestation analysis, and smart city planning
•	Enhanced urban planning and resilience against recurring disasters
•	Providing tools for rapid infrastructure planning in both conflict-affected and disaster-prone regions
•	cost-efficient AI tools
•	disaster preparedness and response in resource-constrained settings.
•	Contribute to Japan's disaster resilience efforts and post-conflict recovery in war-torn regions globally
•	Urban infrastructure planning aid with insights, optimal infrastructure placement, stuff like that


Some Scattered ideas:
“Urban Infrastructure Planning Using AI | Feature Extraction for Urban Land Use Classification from Satellite Imagery | AI-Assisted Urban Reconstruction and Infrastructure Planning | Land-Use Classification for Post-Conflict Reconstruction Using Satellite Imagery and Lightweight Machine Learning Models | AI-Driven Infrastructure Planning for Disaster Resilience and Urban Development | AI-driven system for infrastructure planning and land-use optimization | land-use classification and AI-assisted planning | AI-Assisted Urban Reconstruction and Infrastructure Planning: Integrating Lightweight Machine Learning and Deep Learning for Post-Conflict and Disaster-Resilient Development | AI-Assisted Land-Use Classification and Damage Detection for Post-Conflict and Disaster-Prone Urban Planning | developing an efficient, scalable framework for automated land-use classification and infrastructure damage detection
Develop a machine learning model to analyze satellite images for key urban features (e.g., roads, buildings, vegetation). The model could predict optimal placements for new infrastructure like hospitals, schools, and evacuation centers, accounting for terrain, climate, and population density. Use satellite imagery and computer vision techniques to map and classify land use dynamically, identifying areas suitable for residential, commercial, and industrial purposes. Identifying and classifying specific land-use types (e.g., residential, industrial, green spaces) within urban settings from high-resolution satellite images, This task is critical for urban planning but is often computationally intensive when using deep learning for large-scale images.”


Iterations of the reseach proposal
Final submitted reseach proposal:
Disaster Resilience: Temporal Damage Analysis and Land-Use Classification Using Sparse Subspace Clustering and U-Net
The increasing frequency of natural and man-made disasters necessitates post-event damage assessment and land-use analysis to enhance disaster resilience. Disaster resilience refers to a city’s ability to analyze, respond to, and recover from disasters, requiring precise insights into land-use patterns and structural damage for efficient recovery.
Current post-disaster models often rely on Deep Learning to process satellite and UAV imagery. While accurate, these methods demand significant computational resources, limiting their use in resource-constrained environments (Lee et al., 2023). We propose creating a computationally efficient hybrid framework combining subspace methods and Deep Learning to deliver rapid, actionable insights such as resource allocation and recovery priorities.
Damage assessment identifies infrastructure damage levels, while land-use analysis provides insights into the spatial distribution of resources and human activity. Combining these tasks offers a comprehensive view of affected areas, enabling targeted recovery strategies and improved urban resilience.
We propose using datasets like Sentinel-2, xBD, and xView2, with preprocessing steps including noise removal, resolution alignment, and multispectral band merging. Sentinel-2 classifies land-use features (e.g., urban, rural, vegetation) using high-resolution imagery (Belgiu & Csillik, 2018). The xBD dataset, annotated for pre- and post-disaster damage assessment, trains the model to classify building damage (Gupta et al., 2019), while xView2 enhances classification granularity. Temporal analysis using first and second difference subspaces tracks damage progression and recovery trends. These outputs integrate with Multi-Criteria Decision Analysis (MCDA), which prioritizes recovery efforts based on factors like damage severity and resource availability. UAV imagery complements satellite data by providing localized, high-resolution insights (Xu et al., 2018).
Given that imagery from satellites and UAVs contains an immense amount of data, we propose employing Sparse Subspace Clustering (SSC) for reducing the dimensionality of high-data imagery while preserving key structural features, making the data compact for U-Net segmentation (Elhamifar et al., 2013). Data can be captured by UAVs and processed by SSC on the ground. Preprocessed data from SSC is fed into U-Net to classify damage and land-use pixel-wise (Ronneberger et al., 2015). Results are represented as geospatial heatmaps showing damage intensity, land-use categories, and temporal trends. MCDA synthesizes these outputs to recommend reconstruction priorities and resource allocation. Combining SSC’s preprocessing efficiency with U-Net’s segmentation precision improves computational efficiency in extracting spatial features. The model will be evaluated using metrics like precision, recall, F1 score, and Intersection over Union (IoU).
Incorporating real-time data from UAVs and IoT sensors can enhance adaptability, combining localized imagery with ground-level data (Erdelj & Natalizio, 2016). Temporal analysis tracks evolving damage patterns, while MCDA synthesizes these insights to guide recovery priorities. Testing will cover diverse geographic contexts, including Japan’s disaster-prone regions and war-torn areas with decimated infrastructure.
The result is a scalable, deployable framework for government agencies and NGOs to improve disaster resilience and recovery. The results from this study can also be applied to infrastructure planning, smart cities, and climate change monitoring.
Lee, G. et al. (2023). WATT-EffNet: Lightweight model for disaster images. IEEE Trans. Geosci. Remote Sens., 61, 1-12.
Belgiu, M. & Csillik, O. (2018). Sentinel-2 cropland mapping using dynamic time warping. Remote Sens. Environ., 204, 509-523.
Gupta, R. et al. (2019). xBD: A dataset for building damage assessment. arXiv:1911.09296.
Xu, Y., Wu, L., Xie, Z., & Chen, Z. (2018). Building extraction in very high-resolution remote sensing imagery using deep learning and guided filters. Remote Sensing, 10(1), 144.
Elhamifar, E., & Vidal, R. (2013). Sparse subspace clustering: Algorithm, theory, and applications. IEEE TPAMI, 35(11), 2765-2781.
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. arXiv:1505.04597.
Erdelj, M., & Natalizio, E. (2016). UAV-assisted disaster management: Applications and issues. ICNC Proc., 1-5.

An interation of the research proposal #1:
“Disaster Resilience: Land-Use Analysis and Damage Classification Using Sparse Subspace Clustering and U-Net (after feedback)
The increasing frequency of natural and man-made disasters necessitates post-event damage assessment and land-use analysis to enhance disaster resilience, especially during reconstruction. Disaster resilience refers to a city's ability to analyze, respond to, and recover from disasters, requiring precise insights into land-use patterns and structural damage for efficient recovery.
Current post-disaster assessment models often rely on Deep Learning to process satellite and aerial images, which, while accurate, demand significant computational resources, limiting their applicability in resource-constrained environments (Lee et al., 2023).
This research proposes a hybrid framework combining lightweight subspace methods with Deep Learning for land-use and damage assessment using satellite and UAV imagery. Utilizing temporal analysis through subspace representations, the framework aims to track changes over time. The goal is to create a computationally efficient model for resource-constrained environments to process satellite and UAV imagery, with the focus on the latter, which can help deliver rapid, actionable insights -such as resource allocation and recovery priorities- to urban planners and deployable tools for reconstruction efforts, enhancing disaster resilience.
Damage assessment identifies infrastructure damage levels, while land-use analysis provides insights in spatial distribution of resources and human activity. Combining these tasks offers a comprehensive view of affected areas, enabling targeted recovery strategies, optimal resource allocation, and improved urban resilience.
We propose using datasets such as Sentinel-2, xBD, and xView2, with preprocessing steps including noise removal, resolution alignment, and multispectral band merging. Sentinel-2 will classify land-use features (e.g., urban, rural, vegetation) using high-resolution multispectral imagery (Belgiu & Csillik, 2018). The xBD dataset, annotated for damage assessment before and after disasters, is used for training and validating the model to identify and classify building damage (Gupta et al., 2019). xView2, with detailed annotations of building damage levels, enhances the granularity of damage classification. Incorporating temporal analysis through first and second difference subspaces enables tracking damage progression and recovery trends over time. This dynamic data integrates with Multi-Criteria Decision Analysis (MCDA) to prioritize recovery efforts based on factors such as damage severity, land use, and resource availability. UAV-acquired imagery complements satellite data for localized insights (Xu et al., 2018).
Given that imagery from satellites and UAVs contains an immense amount of different data, we propose employing Sparse Subspace Clustering (SSC) for dimensionality reduction. It's excellent for reducing the dimensionality of the high-data imagery while preserving key critical structural features. This makes the data more compact and manageable, allowing U-Net to focus on key patterns and perform segmentation more efficiently (Elhamifar et al., 2013). We can perform the SSC’s preprocessing on the UAV, then feed the output into the U-Net model on a server to classify the images pixel-wise, and assess information of damage areas and land-use categories (Ronneberger et al., 2015). Results are represented as geospatial heatmaps, showing damage intensity, land-use categories, and temporal trends. MCDA integrates these outputs to prioritize recovery actions based on spatial and temporal insights, such as identifying zones needing immediate reconstruction or resource allocation. Combining SSC’s efficiency in high-dimensional data preprocessing with U-Net's segmentation precision can offer a promising hybrid to improve computational efficiency in extracting spatial features. The model will be evaluated using metrics like precision, recall, F1 score, and Intersection over Union (IoU).
Incorporating real-time data from UAVs and IoT sensors can enhance adaptability, combining localized imagery with ground-level data (Erdelj & Natalizio, 2016). Temporal analysis with subspace methods tracks evolving damage patterns, while MCDA synthesizes these insights to recommend reconstruction priorities and resource allocation strategies. Testing will cover diverse geographic contexts, including Japan’s disaster-prone regions and war-torn areas with decimated infrastructure.
The result is a scalable, deployable framework for government agencies and NGOs to improve disaster resilience and recovery. The results from this study can also be applied to disaster preparedness, infrastructure planning, smart cities, and climate change monitoring.
Lee, G. et al. (2023). WATT-EffNet: Lightweight model for disaster images. IEEE Trans. Geosci. Remote Sens., 61, 1–12.
Belgiu, M. & Csillik, O. (2018). Sentinel-2 cropland mapping using dynamic time warping. Remote Sens. Environ., 204, 509–523.
Gupta, R. et al. (2019). xBD: A dataset for building damage assessment. arXiv:1911.09296.
Xu, Y., Wu, L., Xie, Z., & Chen, Z. (2018). Building extraction in very high-resolution remote sensing imagery using deep learning and guided filters. Remote Sensing, 10(1), 144.
Elhamifar, E., & Vidal, R. (2013). Sparse subspace clustering: Algorithm, theory, and applications. IEEE TPAMI, 35(11), 2765–2781.
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. arXiv:1505.04597.
Erdelj, M., & Natalizio, E. (2016). UAV-assisted disaster management: Applications and issues. ICNC Proc., 1–5.”

an interation of the research proposal #2:
“Disaster Resilience: Temporal Damage Analysis and Land-Use Classification Using Sparse Subspace Clustering and U-Net (shortened version)
The increasing frequency of natural and man-made disasters necessitates post-event damage assessment and land-use analysis to enhance disaster resilience. Disaster resilience refers to a city's ability to analyze, respond to, and recover from disasters, requiring precise insights into land-use patterns and structural damage for efficient recovery.
Current post-disaster models often rely on Deep Learning to process satellite and UAV imagery. While accurate, these methods demand significant computational resources, limiting their use in resource-constrained environments (Lee et al., 2023). We propose creating a computationally efficient hybrid framework combining subspace methods and Deep Learning to deliver rapid, actionable insights such as resource allocation and recovery priorities.
Damage assessment identifies infrastructure damage levels, while land-use analysis provides insights into the spatial distribution of resources and human activity. Combining these tasks offers a comprehensive view of affected areas, enabling targeted recovery strategies and improved urban resilience.
We propose using datasets like Sentinel-2, xBD, and xView2, with preprocessing steps including noise removal, resolution alignment, and multispectral band merging. Sentinel-2 classifies land-use features (e.g., urban, rural, vegetation) using high-resolution imagery (Belgiu & Csillik, 2018). The xBD dataset, annotated for pre- and post-disaster damage assessment, trains the model to classify building damage (Gupta et al., 2019), while xView2 enhances classification granularity. Temporal analysis using first and second difference subspaces tracks damage progression and recovery trends. These outputs integrate with Multi-Criteria Decision Analysis (MCDA), which prioritizes recovery efforts based on factors like damage severity and resource availability. UAV imagery complements satellite data by providing localized, high-resolution insights (Xu et al., 2018).
Given that imagery from satellites and UAVs contains an immense amount of data, we propose employing Sparse Subspace Clustering (SSC) for reducing the dimensionality of high-data imagery while preserving key critical structure features, making the data compact for U-Net segmentation (Elhamifar et al., 2013). Preprocessed data from SSC is fed into U-Net to classify damage and land-use pixel-wise (Ronneberger et al., 2015). Results are represented as geospatial heatmaps showing damage intensity, land-use categories, and temporal trends. MCDA synthesizes these outputs to recommend reconstruction priorities and resource allocation. Combining SSC’s preprocessing efficiency with U-Net's segmentation precision improves computational efficiency in extracting spatial features. The model will be evaluated using metrics like precision, recall, F1 score, and Intersection over Union (IoU).
Incorporating real-time data from UAVs and IoT sensors can enhance adaptability, combining localized imagery with ground-level data (Erdelj & Natalizio, 2016). Temporal analysis tracks evolving damage patterns, while MCDA synthesizes these insights to guide recovery priorities. Testing will cover diverse geographic contexts, including Japan’s disaster-prone regions and war-torn areas with decimated infrastructure.
The result is a scalable, deployable framework for government agencies and NGOs to improve disaster resilience and recovery. The results from this study can also be applied to infrastructure planning, smart cities, and climate change monitoring.
Lee, G. et al. (2023). WATT-EffNet: Lightweight model for disaster images. IEEE Trans. Geosci. Remote Sens., 61, 1–12.
Belgiu, M. & Csillik, O. (2018). Sentinel-2 cropland mapping using dynamic time warping. Remote Sens. Environ., 204, 509–523.
Gupta, R. et al. (2019). xBD: A dataset for building damage assessment. arXiv:1911.09296.
Xu, Y., Wu, L., Xie, Z., & Chen, Z. (2018). Building extraction in very high-resolution remote sensing imagery using deep learning and guided filters. Remote Sensing, 10(1), 144.
Elhamifar, E., & Vidal, R. (2013). Sparse subspace clustering: Algorithm, theory, and applications. IEEE TPAMI, 35(11), 2765–2781.
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. arXiv:1505.04597.
Erdelj, M., & Natalizio, E. (2016). UAV-assisted disaster management: Applications and issues. ICNC Proc., 1–5.”

an interation of the research proposal #3:

“Disaster Resilience: Land-Use and Damage Classification Using Sparse Subspace Clustering and U-Net
The increasing frequency of natural and man-made disasters necessitates post-event damage assessment and land-use analysis to enhance disaster resilience, especially during reconstruction. Disaster resilience refers to a city's ability to analyze, respond to, and recover from disasters, requiring precise insights into land-use patterns and structural damage for efficient recovery.
Current post-disaster assessment models often rely on Deep Learning methods, which, while accurate, demand significant computational resources, limiting their applicability in resource-constrained environments (Lee et al., 2023).
This research proposes a hybrid framework combining lightweight subspace methods with Deep Learning for land-use and damage assessment using high-resolution satellite imagery. The goal is to create a computationally efficient model for resource-constrained recovery zones, delivering rapid, actionable insights to urban planners and deployable tools for reconstruction efforts, enhancing disaster resilience.
Damage assessment identifies infrastructure impact, while land-use analysis provides insights in spatial distribution of resources and human activity. Combining these tasks offers a comprehensive view of affected areas, enabling targeted recovery strategies, optimal resource allocation, and improved urban resilience.
The study utilizes datasets like Sentinel-2, xBD, and xView2, with preprocessing steps including noise removal, resolution alignment, and multispectral band merging. Sentinel-2 will classify land-use features (e.g., urban, rural, vegetation) using high-resolution multispectral imagery (Belgiu & Csillik, 2018). The xBD dataset, annotated for pre- and post-disaster damage assessment, will train and validate the model with augmentation to improve generalization (Gupta et al., 2019). xView2, annotated with gradual building damage levels, enhances damage classification granularity, while UAV-acquired imagery may complement satellite inputs for localized data (Xu et al., 2018).
Given that imagery from satellites or UAVs contains an immense amount of different data, The model employs Sparse Subspace Clustering (SSC) for dimensionality reduction, processing satellite imagery to preserve key structural features (Elhamifar et al., 2013). This output feeds into the U-Net model, specializing in pixel-wise classification and segmentation, enabling precise identification of damage areas and land-use categories in high-resolution imagery (Ronneberger et al., 2015). For extracting spatial features, SSC and U-Net together form a promising hybrid to improve computational efficiency and interpretability.
This hybrid framework uniquely combines SSC's efficiency in high-dimensional data preprocessing with U-Net's segmentation precision, addressing gaps in current methodologies. The model will be evaluated using metrics like precision, recall, F1 score, and Intersection over Union (IoU).
Incorporating real-time data from UAVs and IoT sensors will enhance adaptability, combining localized imagery with ground-level data (Erdelj & Natalizio, 2016). Testing will cover diverse geographic contexts, including Japan’s disaster-prone regions and war-torn areas with decimated infrastructure.
The result is a scalable, deployable framework for government agencies and NGOs to improve disaster resilience and recovery. The cost-efficient AI tools developed can also be applied to disaster preparedness, infrastructure planning, smart cities, and climate change monitoring.
Lee, G. et al. (2023). WATT-EffNet: Lightweight model for disaster images. IEEE Trans. Geosci. Remote Sens., 61, 1–12.
Belgiu, M. & Csillik, O. (2018). Sentinel-2 cropland mapping using dynamic time warping. Remote Sens. Environ., 204, 509–523.
Gupta, R. et al. (2019). xBD: A dataset for building damage assessment. arXiv:1911.09296.
Xu, Y., Wu, L., Xie, Z., & Chen, Z. (2018). Building extraction in very high-resolution remote sensing imagery using deep learning and guided filters. Remote Sensing, 10(1), 144.
Elhamifar, E., & Vidal, R. (2013). Sparse subspace clustering: Algorithm, theory, and applications. IEEE TPAMI, 35(11), 2765–2781.
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. arXiv:1505.04597.
Erdelj, M., & Natalizio, E. (2016). UAV-assisted disaster management: Applications and issues. ICNC Proc., 1–5.”
An interation of the research proposal #4:
“Disaster Resilience: Land-Use and Damage Classification Using Sparse Subspace Clustering and U-Net
The increasing frequency of natural and man-made disasters necessitates post-event damage assessment and land-use analysis to enhance disaster resilience, especially during reconstruction. Disaster resilience refers to a city's ability to analyze, respond to, and recover from disasters, requiring precise insights into land-use patterns and structural damage for efficient recovery.
Current post-disaster assessment models often rely on Deep Learning methods, which, while accurate, demand significant computational resources, limiting their applicability in resource-constrained environments (Lee et al., 2023).
This research proposes a hybrid framework combining lightweight subspace methods with Deep Learning for land-use and damage classification using high-resolution satellite imagery. The goal is to create a computationally efficient model for resource-constrained recovery zones, delivering rapid, actionable insights to urban planners and deployable tools for reconstruction efforts, enhancing disaster resilience.
Damage assessment identifies infrastructure impact, while land-use analysis provides insights in spatial distribution of resources and human activity. Combining these tasks offers a comprehensive view of affected areas, enabling targeted recovery strategies, optimal resource allocation, and improved urban resilience.
The study utilizes datasets like Sentinel-2, xBD, and xView2, with preprocessing steps including noise removal, resolution alignment, and multispectral band merging. Sentinel-2 will classify land-use features (e.g., urban, rural, vegetation) using high-resolution multispectral imagery (Belgiu & Csillik, 2018). The xBD dataset, annotated for pre- and post-disaster damage assessment, will train and validate the model with augmentation to improve generalization (Gupta et al., 2019). xView2, annotated with gradual building damage levels, enhances damage classification granularity, while UAV-acquired imagery may complement satellite inputs for localized data (Xu et al., 2018).
Given that imagery from satellites or UAVs contains an immense amount of different data, The model employs Sparse Subspace Clustering (SSC) for dimensionality reduction, processing satellite imagery to preserve key structural features (Elhamifar et al., 2013). This output feeds into the U-Net model, specializing in pixel-wise classification and segmentation, enabling precise identification of damage areas and land-use categories in high-resolution imagery (Ronneberger et al., 2015). For extracting spatial features, SSC and U-Net together form a promising hybrid to improve computational efficiency and interpretability.
This hybrid framework uniquely combines SSC's efficiency in high-dimensional data preprocessing with U-Net's segmentation precision, addressing gaps in current methodologies. The model will be evaluated using metrics like precision, recall, F1 score, and Intersection over Union (IoU).
Incorporating real-time data from UAVs and IoT sensors will enhance adaptability, combining localized imagery with ground-level data (Erdelj & Natalizio, 2016). Testing will cover diverse geographic contexts, including Japan’s disaster-prone regions and war-torn areas with decimated infrastructure.
The result is a scalable, deployable framework for government agencies and NGOs to improve disaster resilience and recovery. The cost-efficient AI tools developed can also be applied to disaster preparedness, infrastructure planning, smart cities, and climate change monitoring.
Lee, G. et al. (2023). WATT-EffNet: Lightweight model for disaster images. IEEE Trans. Geosci. Remote Sens., 61, 1–12.
Belgiu, M. & Csillik, O. (2018). Sentinel-2 cropland mapping using dynamic time warping. Remote Sens. Environ., 204, 509–523.
Gupta, R. et al. (2019). xBD: A dataset for building damage assessment. arXiv:1911.09296.
Xu, Y., Wu, L., Xie, Z., & Chen, Z. (2018). Building extraction in very high-resolution remote sensing imagery using deep learning and guided filters. Remote Sensing, 10(1), 144.
Elhamifar, E., & Vidal, R. (2013). Sparse subspace clustering: Algorithm, theory, and applications. IEEE TPAMI, 35(11), 2765–2781.
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. arXiv:1505.04597.
Erdelj, M., & Natalizio, E. (2016). UAV-assisted disaster management: Applications and issues. ICNC Proc., 1–5.”
An interation of the research proposal #5:
“Disaster Resilience: Temporal Damage Analysis and Land-Use Classification Using Sparse Subspace Clustering and U-Net

The increasing frequency of natural and man-made disasters necessitates post-event damage assessment and land-use analysis to enhance disaster resilience. Disaster resilience refers to a city's ability to analyze, respond to, and recover from disasters, requiring precise insights into land-use patterns and structural damage for efficient recovery.
Current post-disaster models often rely on Deep Learning to process satellite and UAV imagery. While accurate, these methods demand significant computational resources, limiting their use in resource-constrained environments (Lee et al., 2023). We propose creating a computationally efficient hybrid framework combining subspace methods and Deep Learning to deliver rapid, actionable insights such as resource allocation and recovery priorities.
Damage assessment identifies infrastructure damage levels, while land-use analysis provides insights into the spatial distribution of resources and human activity. Combining these tasks offers a comprehensive view of affected areas, enabling targeted recovery strategies and improved urban resilience.
We propose using datasets like Sentinel-2, xBD, and xView2, with preprocessing steps including noise removal, resolution alignment, and multispectral band merging. Sentinel-2 classifies land-use features (e.g., urban, rural, vegetation) using high-resolution imagery (Belgiu & Csillik, 2018). The xBD dataset, annotated for pre- and post-disaster damage assessment, trains the model to classify building damage (Gupta et al., 2019), while xView2 enhances classification granularity. Temporal analysis using first and second difference subspaces tracks damage progression and recovery trends. These outputs integrate with Multi-Criteria Decision Analysis (MCDA), which prioritizes recovery efforts based on factors like damage severity and resource availability. UAV imagery complements satellite data by providing localized, high-resolution insights (Xu et al., 2018).
Given that imagery from satellites and UAVs contains an immense amount of data, we propose employing Sparse Subspace Clustering (SSC) for reducing the dimensionality of high-data imagery while preserving key structural features, making the data compact for U-Net segmentation (Elhamifar et al., 2013). Data can be captured by UAVs and processed by SSC on the ground. Preprocessed data from SSC is fed into U-Net to classify damage and land-use pixel-wise (Ronneberger et al., 2015). Results are represented as geospatial heatmaps showing damage intensity, land-use categories, and temporal trends. MCDA synthesizes these outputs to recommend reconstruction priorities and resource allocation. Combining SSC’s preprocessing efficiency with U-Net's segmentation precision improves computational efficiency in extracting spatial features. The model will be evaluated using metrics like precision, recall, F1 score, and Intersection over Union (IoU).
Incorporating real-time data from UAVs and IoT sensors can enhance adaptability, combining localized imagery with ground-level data (Erdelj & Natalizio, 2016). Temporal analysis tracks evolving damage patterns, while MCDA synthesizes these insights to guide recovery priorities. Testing will cover diverse geographic contexts, including Japan’s disaster-prone regions and war-torn areas with decimated infrastructure.
The result is a scalable, deployable framework for government agencies and NGOs to improve disaster resilience and recovery. The results from this study can also be applied to infrastructure planning, smart cities, and climate change monitoring.
Lee, G. et al. (2023). WATT-EffNet: Lightweight model for disaster images. IEEE Trans. Geosci. Remote Sens., 61, 1-12.
Belgiu, M. & Csillik, O. (2018). Sentinel-2 cropland mapping using dynamic time warping. Remote Sens. Environ., 204, 509-523.
Gupta, R. et al. (2019). xBD: A dataset for building damage assessment. arXiv:1911.09296.
Xu, Y., Wu, L., Xie, Z., & Chen, Z. (2018). Building extraction in very high-resolution remote sensing imagery using deep learning and guided filters. Remote Sensing, 10(1), 144.
Elhamifar, E., & Vidal, R. (2013). Sparse subspace clustering: Algorithm, theory, and applications. IEEE TPAMI, 35(11), 2765-2781.
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. arXiv:1505.04597.
Erdelj, M., & Natalizio, E. (2016). UAV-assisted disaster management: Applications and issues. ICNC Proc., 1-5.”

a very long iteration for the research proposal #6:
“Land-Use and Damage Classification with Lightweight Subspace and Deep Learning Methods for Disaster Resilience and Urban Planning
1. Abstract
The increasing frequency of natural disasters and urban challenges highlights the need for efficient post-event damage assessment and land-use analysis. This research aims to develop a hybrid classification framework leveraging lightweight subspace methods and deep learning architectures to analyze high-resolution satellite imagery. Focusing on land-use and damage classification, the proposed framework emphasizes computational efficiency and adaptability, enabling its application in resource-constrained environments.
Datasets such as Sentinel-2 and EuroSAT provide a foundation for training and testing the models, while Japan's disaster-prone regions and war-torn areas in other countries offer validation opportunities. The expected outcomes include a scalable model for disaster recovery efforts, insights for urban planners, and a deployable tool to improve disaster resilience. This study bridges the gap between academic research and practical applications, aligning with the needs of post-disaster reconstruction and proactive urban planning globally.

2. Introduction
Context and Motivation
Disasters—both natural and human-induced—pose significant challenges to urban resilience and infrastructure planning. Efficient recovery and reconstruction require rapid and precise insights into land-use patterns and structural damage. For disaster-prone countries like Japan, rebuilding after events such as earthquakes and tsunamis underscores the importance of optimized, data-driven approaches. Similarly, in war-torn regions, where infrastructure is often decimated, the demand for scalable tools to assist in reconstruction is critical.
Problem Statement
Current methods for land-use and damage classification rely heavily on deep learning techniques, which, while accurate, are computationally intensive. These limitations make them less practical in resource-constrained scenarios, such as post-disaster zones or developing regions. There is a pressing need for models that balance computational efficiency and accuracy while remaining adaptable to diverse environments.

Research Gap
Existing solutions often treat damage assessment and land-use analysis as independent tasks. Moreover, many rely on high-end computational setups, limiting their deployment. Few studies integrate classical techniques like subspace methods with modern deep learning frameworks to create lightweight, versatile models.
Proposed Solution
This research proposes a hybrid framework combining subspace and deep learning methods for land-use and damage classification using high-resolution satellite imagery. The approach prioritizes lightweight computation, enabling on-the-ground application in post-disaster scenarios and urban planning efforts.
Datasets
To ensure feasibility, the proposal will utilize established, publicly accessible datasets such as:
1.	Sentinel-2 Satellite Data:
o	Provides multispectral images with a focus on land use and disaster assessment.
o	High resolution for urban areas and disaster-stricken zones.
2.	EuroSAT Dataset:
o	Built on Sentinel-2, specifically for land-use classification tasks.
3.	xView2 Dataset:
o	Focuses on building damage assessment using high-resolution satellite images.
o	Annotated for pre- and post-disaster imagery.
4.	xBD: Building Damage Dataset
o	high-resolution satellite imagery dataset annotated for pre- and post-disaster damage assessment.
Source: Maxar and CrowdAI.


5.	MiniFrance Dataset
o	A high-resolution dataset annotated for urban and rural land-use classification.
Source: Research community.
6.	Urban Atlas (Copernicus Program)
o	Provides standardized high-resolution land-use and land-cover data for European urban areas, ideal for urban reconstruction studies. Source: European Union Copernicus Program.
7.	Planet Labs Data (Potential):
o	Offers daily satellite imagery for specific regions.

3. Objectives
The primary objectives of this research are to:
1.	Develop a Hybrid Model: Combine subspace methods and deep learning techniques for efficient and accurate land-use and damage classification using satellite imagery.
2.	Optimize for Resource-Constrained Environments: Create a lightweight computational model suitable for deployment in post-disaster zones or regions with limited computational resources.
3.	Validate Across Diverse Scenarios: Test the model on datasets covering both natural and human-induced disaster scenarios, including regions in Japan and global war-torn areas.
4.	Facilitate Scalable Applications: Provide a deployable framework for urban planners, disaster relief agencies, and researchers to improve resilience and recovery efforts.
5.	Develop a hybrid machine learning framework leveraging subspaces and CNNs for efficient land-use and damage classification.
6.	Optimize lightweight models for use in computationally constrained settings, making them accessible for real-world applications.
7.	Validate the framework on diverse datasets to ensure applicability to both post-conflict and disaster scenarios.

4. Methodology
1. Data Collection and Preprocessing
•	Leverage publicly available datasets such as Sentinel-2, EuroSAT, and xView2 for land-use and damage classification tasks.
•	Perform preprocessing tasks including normalization, augmentation, and subsetting to handle class imbalance or enhance feature extraction.
2. Model Development
•	Subspace Techniques: Implement Principal Component Analysis (PCA) and Sparse Subspace Clustering (SSC) for dimensionality reduction and feature selection.
•	Deep Learning Component:
o	Use Convolutional Neural Networks (CNNs) for extracting spatial features.
o	Incorporate Transfer Learning with models like ResNet or MobileNet for improved accuracy.
•	Hybrid Integration: Combine the outputs of subspace methods and CNNs using ensemble techniques or hybrid pipelines.
3. Evaluation
•	Performance Metrics:
o	Accuracy, Precision, Recall, and F1-score for classification.
o	Computational efficiency in terms of runtime and memory usage.
•	Compare against baseline models (e.g., standalone CNNs, traditional classifiers) on benchmark datasets.
4. Deployment Feasibility
•	Explore lightweight deployment options using TensorFlow Lite or PyTorch Mobile for edge devices.
•	Simulate real-world scenarios using disaster-stricken regions' datasets.

(Refine methodology based on early experimentation details. For instance:)
•	Use lightweight subspace methods (e.g., PCA) for feature extraction combined with CNNs (e.g., MobileNet) for classification tasks.
•	Train models on openly available datasets like Sentinel-2 for land-use mapping and xBD for disaster damage classification.
•	Evaluate on metrics such as classification accuracy, computational efficiency, and model adaptability to different geographic contexts.
•	Incorporate preprocessing pipelines for multispectral imagery, ensuring compatibility across datasets.


5. Expected Outcomes
1.	Technical Contributions:
o	A novel hybrid framework for land-use and damage classification.
o	Open-source codebase for reproducibility and adaptability.
2.	Societal Impact:
o	Improved disaster response and recovery planning, benefiting communities in Japan and beyond.
o	Scalable tools for urban planners in both developed and developing regions.
3.	Academic Relevance:
o	Insights into the integration of classical and modern machine learning methods(?).
o	Validation of lightweight models for large-scale remote sensing tasks.
4.	Others:
o	Contribute to Japan's disaster resilience efforts and post-conflict recovery in war-torn regions globally.

6. Feasibility and Resources
1.	Datasets: Publicly available, covering diverse disaster scenarios and land-use types, ensuring wide applicability.
2.	Technical Tools:
o	Frameworks: TensorFlow, PyTorch.
o	High-performance computing resources (available through your lab).
3.	Timeline:
o	Data Collection and Preprocessing: 1-2 months.
o	Model Development and Testing: 4-5 months.
o	Validation and Documentation: 2-3 months.

7. Broader Implications
1.	For Japan:
o	Application in earthquake and tsunami recovery efforts.
o	Enhanced urban planning and resilience against recurring disasters.
2.	For Global Context:
o	Post-conflict urban reconstruction using cost-efficient AI tools.
o	Bridging gaps in disaster preparedness and response in resource-constrained settings.
highlight the potential of combining classical and modern machine learning techniques for high-resolution satellite imagery analysis. Here's a synthesis of key findings and resources to validate the research:
Relevant Research and Datasets
1.	Hybrid Techniques: Several studies have explored the integration of classical methods (e.g., subspaces) with deep learning architectures like CNNs and U-Nets for land-use and damage classification. For example:
o	SegNet and U-Net Models: These have been utilized for pixel-level classification, offering high accuracy in delineating land use and cover types such as vegetation, buildings, and water bodies using high-resolution imagery【66】【67】.
o	Subspace-Based Methods: These are effective in reducing dimensionality and capturing structural patterns in large datasets, which can complement deep learning models by improving computational efficiency and interpretability【65】【66】.

2.	Datasets:
o	Open datasets such as those from Sentinel-2, PlanetScope, and UAV-acquired imagery have been frequently used in urban planning and disaster resilience studies. These datasets provide high-resolution imagery essential for training and validating models【67】.
o	Annotated datasets, like those used in competitions (e.g., the Big Data and Computing Intelligence Contest), offer categorized and processed data for urban and rural regions, making them suitable for land-use classification tasks【67】.
3.	Evaluation Metrics:
o	Studies emphasize the importance of metrics such as F1-score, Intersection over Union (IoU), and visual evaluations to assess model performance comprehensively【67】.
Objectives Refined for Feasibility
•	Develop a hybrid framework combining subspace methods for feature extraction and CNN-based models (e.g., SegNet, U-Net) for detailed classification.
•	Focus on a scalable application for both post-conflict and disaster-affected areas, providing actionable maps for land-use and damage.
•	Evaluate on existing datasets and validate on localized regions such as Japan's earthquake-prone areas and your home country's conflict zones.
Future Perspectives and Contributions
This research can generalize to address global challenges:
•	In Japan: Optimizing urban rebuilding strategies post-natural disasters like earthquakes and tsunamis.
•	Globally: Providing tools for rapid infrastructure planning in both conflict-affected and disaster-prone regions.
•	Contribution: Bridging the gap between classical ML efficiency and deep learning precision, creating cost-effective and interpretable solutions for real-world applications.
•	Adapt the methodology for use in other domains, such as climate change monitoring, deforestation analysis, and smart city planning.
•	 Enhance the framework with real-time data from drones or IoT sensors for dynamic infrastructure planning.
•	Expand collaborations to include government agencies, NGOs, and international research institutions to scale the impact of the research.

•	Sekiya, T., et al. (2020). "A Hybrid Subspace Learning Method for Damage Detection in Satellite Imagery." Remote Sensing, 12(9).
•	Zhang, L., et al. (2021). "Deep Learning for Satellite Image Classification: A Review of Techniques and Applications." IEEE Access, 9.
•	Li, Y., et al. (2022). "Application of Convolutional Neural Networks in Disaster Recovery Using Satellite Data." Sensors, 22(6).
•	Xie, Z., et al. (2019). "Multi-source Remote Sensing Data for Urban Planning: Integration of Sentinel-2 and UAVs." Journal of Urban Technology, 26(1).
•	Kim, J., et al. (2023). "PCA-based Subspace Methods for Remote Sensing Data Processing in Urban Areas." ISPRS Journal of Photogrammetry, 76(2).
•	Chen, X., et al. (2022). "Urban Resilience Enhancement with Hybrid Machine Learning Models." Nature Sustainability, 7(8).
•	Tan, Q., et al. (2020). "Efficient Disaster Management with Lightweight Deep Learning Models." IEEE Transactions on Geoscience and Remote Sensing, 58(12).
•	Yuan, S., et al. (2021). "Deep Learning for Land Use Classification: Challenges and Solutions." Remote Sensing, 13(5).
•	Smith, P., et al. (2024). "Satellite-based Disaster Response: Hybrid Models for Rapid Classification." Remote Sensing Applications, 25.”
a research proposal iteration with a senpai’s comments #7:
“Disaster Resilience: Land-Use Analysis and Damage Classification Using Sparse Subspace Clustering and U-Net
The increasing frequency of natural and man-made disasters necessitates post-event damage assessment and land-use analysis to enhance disaster resilience, especially during reconstruction. Disaster resilience refers to a city's ability to analyze, respond to, and recover from disasters, requiring precise insights into land-use patterns and structural damage for efficient recovery.
Current post-disaster assessment models often rely on Deep Learning methods, which, while accurate, demand significant computational resources, limiting their applicability in resource-constrained environments (Lee et al., 2023). [senpais comment to this section: I didn't get how deep learning can be used to assess post-disaster anything. Usually what comes to (my) mind on this type of research is using simulation models / ArcGIS. You should explain that you are using DNNs to process aerial images of disaster areas here.]
This research proposes a hybrid framework combining lightweight subspace methods with Deep Learning for land-use and damage assessment using high-resolution satellite imagery(1). The goal is to create a computationally efficient model for resource-constrained environments, which can help deliver rapid, actionable insights to urban planners and deployable tools for reconstruction efforts(2), enhancing disaster resilience. [senpais comment to this section: (1) If you are using satellite imagery, why a computationally efficient model is necessary? Won't this run on a server somewhere? on Lee et. al. they run their software on UAV, so the computational efficiency seems to make more sense. (2) This is a bit abstract. How exactly does a more computationally efficient model can help with disaster resilience? Can't current models already do that?]
Damage assessment identifies infrastructure impact, while land-use analysis provides insights in spatial distribution of resources and human activity. Combining these tasks offers a comprehensive view of affected areas, enabling targeted recovery strategies, optimal resource allocation, and improved urban resilience.
We propose using datasets such as Sentinel-2, xBD, and xView2, with preprocessing steps including noise removal, resolution alignment, and multispectral band merging. Sentinel-2 will classify land-use features (e.g., urban, rural, vegetation) using high-resolution multispectral imagery (Belgiu & Csillik, 2018). The xBD dataset, annotated for pre and post-disaster damage assessment, is used for…will train and validate the model with augmentation to improve generalization (Gupta et al., 2019). xView2, annotated with gradual building damage levels, enhances damage classification granularity, while UAV-acquired imagery is used to complement satellite imagery for localized data (Xu et al., 2018).
Given that imagery from satellites and UAVs contains an immense amount of different data, we propose employing Sparse Subspace Clustering (SSC) for dimensionality reduction, processing satellite imagery to preserve key structural features (Elhamifar et al., 2013). We feed this output into the U-Net model to classify the images pixel-wise, and assess information of damage areas and land-use categories (Ronneberger et al., 2015). For extracting spatial features, SSC and U-Net together {form a promising hybrid to improve computational efficiency and interpretability} [senpai’s comment: Why? How does pre-processing with SSC helps UNet? I really don't know, maybe you can ask Gulpi to help you in this, and Why is interpretability important?], by combining SSC’s efficiency… .
This hybrid framework uniquely combines SSC's efficiency in high-dimensional data preprocessing with U-Net's segmentation precision. The model will be evaluated using metrics like precision, recall, F1 score, and Intersection over Union (IoU).
Additionally, Incorporating real-time data from UAVs and IoT sensors can enhance adaptability, combining localized imagery with ground-level data (Erdelj & Natalizio, 2016). Testing will cover diverse geographic contexts, including Japan’s disaster-prone regions and war-torn areas with decimated infrastructure.
The result is a scalable, deployable framework for government agencies and NGOs to improve disaster resilience and recovery. The results from this study can also be applied to disaster preparedness, infrastructure planning, smart cities, and climate change monitoring.
Lee, G. et al. (2023). WATT-EffNet: Lightweight model for disaster images. IEEE Trans. Geosci. Remote Sens., 61, 1–12.
Belgiu, M. & Csillik, O. (2018). Sentinel-2 cropland mapping using dynamic time warping. Remote Sens. Environ., 204, 509–523.
Gupta, R. et al. (2019). xBD: A dataset for building damage assessment. arXiv:1911.09296.
Xu, Y., Wu, L., Xie, Z., & Chen, Z. (2018). Building extraction in very high-resolution remote sensing imagery using deep learning and guided filters. Remote Sensing, 10(1), 144.
Elhamifar, E., & Vidal, R. (2013). Sparse subspace clustering: Algorithm, theory, and applications. IEEE TPAMI, 35(11), 2765–2781.
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. arXiv:1505.04597.
Erdelj, M., & Natalizio, E. (2016). UAV-assisted disaster management: Applications and issues. ICNC Proc., 1–5.”

An the iteration before the final one #8:
“Disaster Resilience: Land-Use and Damage Classification Using Sparse Subspace Clustering and U-Net
The increasing frequency of natural and man-made disasters necessitates post-event damage assessment and land-use analysis to enhance disaster resilience, especially during reconstruction. Disaster resilience refers to a city's ability to analyze, respond to, and recover from disasters, requiring precise insights into land-use patterns and structural damage for efficient recovery.
Current post-disaster assessment models often rely on Deep Learning methods, which, while accurate, demand significant computational resources, limiting their applicability in resource-constrained environments (Lee et al., 2023).
This research proposes a hybrid framework combining lightweight subspace methods with Deep Learning for land-use and damage classification using high-resolution satellite imagery. The goal is to create a computationally efficient model for resource-constrained recovery zones, delivering rapid, actionable insights to urban planners and deployable tools for reconstruction efforts, enhancing disaster resilience.
Damage assessment identifies infrastructure impact, while land-use analysis provides insights in spatial distribution of resources and human activity. Combining these tasks offers a comprehensive view of affected areas, enabling targeted recovery strategies, optimal resource allocation, and improved urban resilience.
The study utilizes datasets like Sentinel-2, xBD, and xView2, with preprocessing steps including noise removal, resolution alignment, and multispectral band merging. Sentinel-2 will classify land-use features (e.g., urban, rural, vegetation) using high-resolution multispectral imagery (Belgiu & Csillik, 2018). The xBD dataset, annotated for pre- and post-disaster damage assessment, will train and validate the model with augmentation to improve generalization (Gupta et al., 2019). xView2, annotated with gradual building damage levels, enhances damage classification granularity, while UAV-acquired imagery may complement satellite inputs for localized data (Xu et al., 2018).
Given that imagery from satellites or UAVs contains an immense amount of different data, The model employs Sparse Subspace Clustering (SSC) for dimensionality reduction, processing satellite imagery to preserve key structural features (Elhamifar et al., 2013). This output feeds into the U-Net model, specializing in pixel-wise classification and segmentation, enabling precise identification of damage areas and land-use categories in high-resolution imagery (Ronneberger et al., 2015). For extracting spatial features, SSC and U-Net together form a promising hybrid to improve computational efficiency and interpretability.
This hybrid framework uniquely combines SSC's efficiency in high-dimensional data preprocessing with U-Net's segmentation precision, addressing gaps in current methodologies. The model will be evaluated using metrics like precision, recall, F1 score, and Intersection over Union (IoU).
Incorporating real-time data from UAVs and IoT sensors will enhance adaptability, combining localized imagery with ground-level data (Erdelj & Natalizio, 2016). Testing will cover diverse geographic contexts, including Japan’s disaster-prone regions and war-torn areas with decimated infrastructure.
The result is a scalable, deployable framework for government agencies and NGOs to improve disaster resilience and recovery. The cost-efficient AI tools developed can also be applied to disaster preparedness, infrastructure planning, smart cities, and climate change monitoring.
Lee, G. et al. (2023). WATT-EffNet: Lightweight model for disaster images. IEEE Trans. Geosci. Remote Sens., 61, 1–12.
Belgiu, M. & Csillik, O. (2018). Sentinel-2 cropland mapping using dynamic time warping. Remote Sens. Environ., 204, 509–523.
Gupta, R. et al. (2019). xBD: A dataset for building damage assessment. arXiv:1911.09296.
Xu, Y., Wu, L., Xie, Z., & Chen, Z. (2018). Building extraction in very high-resolution remote sensing imagery using deep learning and guided filters. Remote Sensing, 10(1), 144.
Elhamifar, E., & Vidal, R. (2013). Sparse subspace clustering: Algorithm, theory, and applications. IEEE TPAMI, 35(11), 2765–2781.
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. arXiv:1505.04597.
Erdelj, M., & Natalizio, E. (2016). UAV-assisted disaster management: Applications and issues. ICNC Proc., 1–5.”

last iteration of the resaerch proposal #9:
“Disaster Resilience: Temporal Damage Analysis and Land-Use Classification Using Sparse Subspace Clustering and U-Net
The increasing frequency of natural and man-made disasters necessitates post-event damage assessment and land-use analysis to enhance disaster resilience. Disaster resilience refers to a city’s ability to analyze, respond to, and recover from disasters, requiring precise insights into land-use patterns and structural damage for efficient recovery.
Current post-disaster models often rely on Deep Learning to process satellite and UAV imagery. While accurate, these methods demand significant computational resources, limiting their use in resource-constrained environments (Lee et al., 2023). We propose creating a computationally efficient hybrid framework combining subspace methods and Deep Learning to deliver rapid, actionable insights such as resource allocation and recovery priorities.
Damage assessment identifies infrastructure damage levels, while land-use analysis provides insights into the spatial distribution of resources and human activity. Combining these tasks offers a comprehensive view of affected areas, enabling targeted recovery strategies and improved urban resilience.
We propose using datasets like Sentinel-2, xBD, and xView2, with preprocessing steps including noise removal, resolution alignment, and multispectral band merging. Sentinel-2 classifies land-use features (e.g., urban, rural, vegetation) using high-resolution imagery (Belgiu & Csillik, 2018). The xBD dataset, annotated for pre- and post-disaster damage assessment, trains the model to classify building damage (Gupta et al., 2019), while xView2 enhances classification granularity. Temporal analysis using first and second difference subspaces tracks damage progression and recovery trends. These outputs integrate with Multi-Criteria Decision Analysis (MCDA), which prioritizes recovery efforts based on factors like damage severity and resource availability. UAV imagery complements satellite data by providing localized, high-resolution insights (Xu et al., 2018).
Given that imagery from satellites and UAVs contains an immense amount of data, we propose employing Sparse Subspace Clustering (SSC) for reducing the dimensionality of high-data imagery while preserving key structural features, making the data compact for U-Net segmentation (Elhamifar et al., 2013). Data can be captured by UAVs and processed by SSC on the ground. Preprocessed data from SSC is fed into U-Net to classify damage and land-use pixel-wise (Ronneberger et al., 2015). Results are represented as geospatial heatmaps showing damage intensity, land-use categories, and temporal trends. MCDA synthesizes these outputs to recommend reconstruction priorities and resource allocation. Combining SSC’s preprocessing efficiency with U-Net’s segmentation precision improves computational efficiency in extracting spatial features. The model will be evaluated using metrics like precision, recall, F1 score, and Intersection over Union (IoU).
Incorporating real-time data from UAVs and IoT sensors can enhance adaptability, combining localized imagery with ground-level data (Erdelj & Natalizio, 2016). Temporal analysis tracks evolving damage patterns, while MCDA synthesizes these insights to guide recovery priorities. Testing will cover diverse geographic contexts, including Japan’s disaster-prone regions and war-torn areas with decimated infrastructure.
The result is a scalable, deployable framework for government agencies and NGOs to improve disaster resilience and recovery. The results from this study can also be applied to infrastructure planning, smart cities, and climate change monitoring.
Lee, G. et al. (2023). WATT-EffNet: Lightweight model for disaster images. IEEE Trans. Geosci. Remote Sens., 61, 1-12.
Belgiu, M. & Csillik, O. (2018). Sentinel-2 cropland mapping using dynamic time warping. Remote Sens. Environ., 204, 509-523.
Gupta, R. et al. (2019). xBD: A dataset for building damage assessment. arXiv:1911.09296.
Xu, Y., Wu, L., Xie, Z., & Chen, Z. (2018). Building extraction in very high-resolution remote sensing imagery using deep learning and guided filters. Remote Sensing, 10(1), 144.
Elhamifar, E., & Vidal, R. (2013). Sparse subspace clustering: Algorithm, theory, and applications. IEEE TPAMI, 35(11), 2765-2781.
Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional networks for biomedical image segmentation. arXiv:1505.04597.
Erdelj, M., & Natalizio, E. (2016). UAV-assisted disaster management: Applications and issues. ICNC Proc., 1-5.”

References:
•	https://eleks.com/research/deep-learning-for-damage-detection-using-satellite-images/
•	https://arxiv.org/abs/1505.04597
•	https://arxiv.org/abs/1801.04381
•	https://www.mdpi.com/2072-4292/16/12/2177
•	https://www.sciencedirect.com/science/article/abs/pii/S0957417424015288
•	https://arxiv.org/abs/2302.13028
•	https://ieeexplore.ieee.org/document/9463715
•	https://www.sciencedirect.com/science/article/pii/S1569843223003771
•	https://arxiv.org/html/2403.11735v4
•	https://www.researchgate.net/publication/378405460_GARP_-_A_Hybrid_Preprocessing_Technique_for_Semantic_Segmentation_of_Satellite_Images_with_U-Net_Architecture
•	https://isprs-archives.copernicus.org/articles/XLII-1-W2/13/2019/
•	https://www.mdpi.com/2072-4292/14/9/1977
•	https://www.researchgate.net/publication/370669231_Land_Use_and_Land_Cover_Mapping_with_VHR_and_Multi-Temporal_Sentinel-2_Imagery
•	https://www.researchgate.net/publication/381116177_Deep_Learning_for_Satellite_Image_Segmentation_Deforestation_Detection_and_Reforestation_Zone_Identification
•	https://www.mdpi.com/2072-4292/15/10/2501
•	https://link.springer.com/article/10.1007/s12145-023-01205-2
•	https://ieeexplore.ieee.org/abstract/document/10197573
•	https://ieeexplore.ieee.org/abstract/document/8605374
•	https://www.tandfonline.com/doi/full/10.1080/01431161.2024.2379515
•	https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2022.969758/full
•	https://www.sciencedirect.com/science/article/pii/S1470160X24005247
•	https://www.sciencedirect.com/science/article/abs/pii/S1474706522001772
•	https://www.bing.com/search?pc=OA1&q=hybrid%20models%20for%20disaster%20resilience%20using%20satellite%20imagery%20land%20use%20classification%20subspace%20and%20CNN
•	https://www.nature.com/articles/s41598-024-67186-4
•	https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0300473
•	https://ieeexplore.ieee.org/document/9274417
•	https://www.mdpi.com/2072-4292/14/19/4978
•	https://ieeexplore.ieee.org/document/10537666
•	https://www.mdpi.com/1424-8220/21/23/8083
•	https://hai.stanford.edu/sites/default/files/2024-07/HAI-Policy-Brief-Using-AI-Map-Urban-Change.pdf
•	https://www.euspaceimaging.com/blog/2021/02/18/from-urban-to-rural-enabling-sustainable-urban-planning-and-development-using-satellite-imagery/
•	https://www.semanticscholar.org/search?q=Remote%20sensing%20survey%20infrastructure%20machine%20learning%20challanges&sort=relevance
•	https://www.sciencedirect.com/science/article/abs/pii/S0924271623001582?via%3Dihub
•	https://spaceknow.com/blog/satellites-ai-urbanization-a-modern-approach-to-urban-planning-and-monitoring/
•	https://earthi.space/blog/future-urban-planning/
•	https://github.com/satellite-image-deep-learning/techniques
•	https://link.springer.com/chapter/10.1007/978-981-10-5903-2_34
•	https://www.mdpi.com/2072-4292/16/16/3024
•	https://www.mdpi.com/journal/remotesensing/special_issues/O14L387P95
•	https://www.mdpi.com/journal/jimaging
•	https://www.mdpi.com/journal/jimaging/special_issues/4XJ43H65X2#Research
•	https://www.sciencedirect.com/science/article/abs/pii/S0924271623001582
•	https://www.sciencedirect.com/journal/isprs-journal-of-photogrammetry-and-remote-sensing
•	https://link.springer.com/chapter/10.1007/978-981-10-5903-2_34
•	https://arxiv.org/abs/1911.09296
•	https://arxiv.org/abs/1910.06444

