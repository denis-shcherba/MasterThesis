# Literature

## Papers on Policy Distillation

### Learning by cheating [[link]](https://arxiv.org/pdf/1912.12294)
- Original paper introducing a novel type of policy distillation with the student teacher architecture


### Continual Policy Distillation of Reinforcement Learning-based Controllers for Soft Robotic In-Hand Manipulation [[link]](https://arxiv.org/pdf/2404.04219)

- Leverages Policy Distillation, Continual Learning and Soft Robotics

### UniGraspTransformer [[link]](https://arxiv.org/abs/2412.02699)

- Novel paper on Policy distillation for manipulation


### Learning by Watching: Physical Imitation of Manipulation Skills from Human Videos [[link]](https://arxiv.org/pdf/2101.07241)


## Papers on Robotic Manipulation in Cluttered Environments

### Towards robustly picking unseen objects from densely packed shelves [[link]](https://robotic-manipulation.sciencehub.uw.edu/static/preprints/2023-grotz_rss.pdf)

- Automating object picking in industrial warehouses, specifically for previously unseen objects in densely packed shelves (Also using Amazon shelf)

- Visual segmentation and tracking of unseen objects in cluttered environments are listed as **Challenges**

- Focus on Manipulation planning and control for picking objects without disrupting the shelf arrangement.

- System Components: Shelf inventory tracking, Object re-identification, Autonomous picking

![Pipeline](images/towardsRobustlyPickingPipeline.png)

### Efficient push-grasping for multiple target objects in clutter environments [[link]](https://www.frontiersin.org/journals/neurorobotics/articles/10.3389/fnbot.2023.1188468/full) 

- method proposes a "push-grasping" method that integrates pushing actions to create more space for grasping, thereby reducing the total number of actions required to complete the task.a

### Sim-Grasp: Learning 6-DOF Grasp Policies for Cluttered Environments Using a Synthetic Benchmark [[link]](https://arxiv.org/abs/2405.00841)
- This paper presents Sim-Grasp, a framework for learning 6-degree-of-freedom grasp policies in highly cluttered environments. Using a synthetic benchmark, the authors evaluate different grasping strategies and highlight the importance of simulation-driven learning for real-world applications.

### Hierarchical Visual Policy Learning for Long-Horizon Robot Manipulation in Densely Cluttered Scenes [[link]](https://arxiv.org/abs/2312.02697)
-  The authors propose a hierarchical visual policy learning approach to enable robots to perform long-horizon manipulation tasks in cluttered scenes. By integrating visual perception and reinforcement learning, the system improves efficiency in real-world applications.

### Review of Learning-Based Robotic Manipulation in Cluttered Environments [[link]](https://www.mdpi.com/1424-8220/22/20/7938)
-   Review paper of late **August 2022** surveys of recent advancements in learning-based robotic manipulation for cluttered environments. Categorizes approaches into object removal, assembly, rearrangement, and retrieval.

### Technological development and optimization of pushing and grasping functions in robot arms: A review [[link]](https://www.sciencedirect.com/science/article/pii/S0263224124016142)
- **2024** Review paper of approaches that combine pushing and grasping, particularly to enable grasps of cluttered or stacked objects
### CEPB dataset: a photorealistic dataset to foster the research on bin picking in cluttered environments [[link]](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2024.1222465/full)
-  Self explanatory dataset paper for cluttered environments

## Papers on Force Proprioception-based Policies 

### FoAR: Force-Aware Reactive Policy for Contact-Rich Robotic Manipulation [[link]](https://arxiv.org/pdf/2411.15753l)


![Pipeline](images/foarPipeline.png)

### Precise Object Placement Using Force-Torque Feedback [[link]](https://arxiv.org/pdf/2404.17668)

- 2024 workshop paper

### PROPRIOCEPTIVE LEARNING WITH SOFT POLYHEDRAL NETWORKS [[link]](https://arxiv.org/pdf/2308.08538)
- 2024, TODO

### Fingertip 6-Axis Force/Torque Sensing for Texture Recognition in Robotic Manipulation [[link]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9613688)
- 2021 paper, interesting if we want to discriminate books based on their texture?

### Reinforcement Learning on Variable Impedance Controller for High-Precision Robotic Assembly [[link]](https://arxiv.org/pdf/1903.01066)
- 2019 paper, introduced RL with Force Sensing


### Learning Force Control Policies for Compliant Manipulation [[link]](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=6095096)

- 2011 Schaal classical Paper on using Force Control for manipulation
