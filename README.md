<h1>HAREDNet: A Deep Learning based Architecture for Autonomous Video Surveillance by Recognizing Human Actions</h1>

<p>In this article, a hybrid recognition technique called HAREDNet is proposed, which has a) Encoder-Decoder Network (EDNet) to extract deep features; b) improved Scale-Invariant Feature Transform (iSIFT), improved Gabor (iGabor) and Local Maximal Occurrence (LOMO) techniques to extract local features; c) Cross-view Quadratic Discriminant Analysis (CvQDA) algorithm to reduce the feature redundancy; and d) weighted fusion strategy to merge properties of different essential features.</p>
<p>The proposed technique is evaluated on three (3) publicly available datasets, including NTU RGB+D, HMDB51, and UCF-101, and achieved average recognition accuracy of 97.45%, 80.58%, and 97.48%, respectively, which is better than previously proposed methods.</p>

<h3>Graphical Abstract</h3>
<img src="https://user-images.githubusercontent.com/122672521/217162996-db4a4ba5-07d2-4c70-b7d0-3386d1bd1e5c.jpg" alt="Proposed HAREDNet model">

<h3>Proposed HAREDNet model</h3>
<img src="https://user-images.githubusercontent.com/122672521/217162999-d99fb53d-f0ec-4bf1-bcb0-6cb93c41655d.jpg" alt="Proposed HAREDNet model">

<h3>Intra-class variations and viewpoint variations in a single class (first two rows show intra-class variations while last two rows show different viewpoint variations for a single class)</h3>
<img src="https://user-images.githubusercontent.com/122672521/217152713-95d328eb-e465-4fd9-9581-49a24eb32f7f.jpg" alt="Intra-class variations and viewpoint variations in a single class">

<h3>Procedure of extracting LOMO features</h3>
<img src="https://user-images.githubusercontent.com/122672521/217163002-d9df352e-3195-48de-a74e-b8141bfcc5ab.jpg" alt="Procedure of extracting LOMO features">

<h3>Sample images from selected dataset. First row D1: (left-to-right: writing, giving something to other person (gstop), punching, kicking other person (kop)); second row: D2 (left-to-right: ride_bike, drink, brush_hair, situp); and third row D3: (left-to-right: PlayingDhol (pd), JumpRope (jr), BoxingPunchingBag (bpb), HeadMassage (hm))</h3>
<img src="https://user-images.githubusercontent.com/122672521/217163003-445e4342-6bcd-48ca-b7b6-0faa357219f9.jpg" alt="Sample images from selected dataset">

<h3>Correctly labeled data using HAREDNet</h3>
<img src="https://user-images.githubusercontent.com/122672521/217162987-1a9ab145-b1a7-491e-b6fe-ec4bbe575898.jpg" alt="Correctly labeled data using HAREDNet">

<h3>Wrong predictions using HAREDNet</h3>
<img src="https://user-images.githubusercontent.com/122672521/217163004-d5e82b74-8d00-47cf-8cee-5a97999a7608.jpg" alt="Wrong predictions using HAREDNet">


<h3>Citation</h3>
<h3>Nasir, Inzamam Mashood, et al. "HAREDNet: A deep learning based architecture for autonomous video surveillance by recognizing human actions." Computers and Electrical Engineering 99 (2022): 107805.</h3>
