# ML---Alchohol-Effect-In-Student-Prediction
Alcohol use is a well-known risk factor for certain conditions, such as cirrhosis of the liver, fetal alcohol syndrome, and injuries related to drunk driving. Alcohol consumption can also contribute to chronic illnesses like heart disease, stroke, and some cancers. Interpersonal violence, self-harm (suicide), and unintentional injuries can also be fueled by alcohol. Alcohol has been used for various purposes in many human societies for over ten thousand years (Smart, 2007). The aim of this work is to analyze the effect of alcohol consumption on the academic performance of students of Adekunle Ajasin University using Data mining. 
<p><strong>CHAPTER FOUR</strong></p>
<p><strong>IMPLEMENTATION</strong></p>
<p><strong>4.1</strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>OVERVIEW OF IMPLEMENTATION</strong></p>
<p>This chapter describes the implementation stages for the development of a clustering model. The clustering algorithm used is K-Means. The modeling was done using eleven metrics to determine the success of this implementation. The model was implemented using WEKA 3.8 and a PC.</p>
<p><strong>4.2</strong>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <strong>IMPLEMENTATION REQUIREMENT</strong></p>
<p>The model development requirements are into two main parts, software and hardware requirements. The candidates&rsquo; credibility datasets form a major component for the model building.</p>
<ol>
<li>Software Requirements
<ol start="3">
<li>WEKA v3.8</li>
<li>Clustering API</li>
</ol>
</li>
<li>Hardware Requirements
<ol>
<li>HP, Intel CORE i3, 1 GHz processor, 2GB RAM, 64-bits OS</li>
</ol>
</li>
</ol>
<p><strong>&nbsp;</strong></p>
<p><strong>4.2.1</strong>&nbsp;&nbsp;&nbsp; <strong>MODEL PARAMETER</strong></p>
<p>The Clustering Model Metrics are outlined below.</p>
<ol>
<li>Name &ndash; Name of the individual.</li>
<li>Gender &ndash; Gender [Male or Female].</li>
<li>Morning &ndash; [0,1] If the candidate takes alcohol in the morning.</li>
<li>Afternoon &ndash; [0,1] If the candidate takes alcohol in the afternoon.</li>
<li>Night &ndash; [0,1] If the candidate takes alcohol in the night.</li>
<li>Exam period &ndash; [0,1] If the candidate takes alcohol during the exam period</li>
<li>1<sup>st</sup> GPA &ndash; This is the first of the last three GPA of the candidate.</li>
<li>2nd GPA &ndash; This is the second of the last three GPA of the candidate.</li>
<li>3rd GPA &ndash; This is the Third of the last three GPA of the candidate.</li>
<li>Total &ndash; This is the summation of the three CPA. (1<sup>st</sup> gpa + 2<sup>nd</sup> gpa + 3<sup>rd</sup> gpa)</li>
<li>Average &ndash; This is the average of the GPA, which is calculated (1<sup>st</sup> gpa + 2<sup>nd</sup> gpa + 3<sup>rd</sup> gpa / 3)</li>
</ol>
<p><strong>&nbsp;</strong></p>
<p><strong>4.3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; IMPLEMENTATION AND DESIGN PHASE</strong></p>
<p>The implementation phase of this study shows how the experiment was carried out.</p>
<p><strong>4.3.1&nbsp;&nbsp;&nbsp; WEKA GUI CHOOSER SCREEN</strong></p>
<p>After the successful launch of WEKA v3.8. The first screen that displays is the chooser screen. The chooser screen display various option. In my case, we select the explorer button. The&nbsp;Weka&nbsp;Knowledge&nbsp;Explorer&nbsp;is an easy to use graphical user interface that harnesses the power of the&nbsp;weka&nbsp;software. Each of the major&nbsp;weka&nbsp;packages</p>
<p>&nbsp;<img src="https://github.com/Way4ward17/ML---Alchohol-Effect-In-Student-Prediction/blob/master/Picture1.png" alt="" width="386" height="386" /></p>
<p><strong>4.3.2&nbsp;&nbsp;&nbsp; DATA PROCESSING SCREEN</strong></p>
<p>The first task in the preprocessing step is importing the dataset to WEKA. The screenshot below shows how the dataset was imported. The dataset used was in a CVS file. I clicked done the Open file button, the chooser dialog box appeared, then I went to the directory where the dataset was saved.</p>
<p><img src="https://github.com/Way4ward17/ML---Alchohol-Effect-In-Student-Prediction/blob/master/Picture2.png" alt="" width="600" height="300" /></p>
<p>After the dataset has been successfully imported. It is possible to view all the whole datasets to see if there are one or more errors in the dataset which can lead to data preparation. After a successful check. There was no error in the dataset imported. But the name isn't an important metric to be considered. I removed the name from the dataset to avoid complexity and more accuracy.</p>
<p><img src="https://github.com/Way4ward17/ML---Alchohol-Effect-In-Student-Prediction/blob/master/Picture3.png" alt="" width="700" height="500" />&nbsp;</p>
<p>The screenshot below shows when the name metric was remove. The remaining dataset would be used for the experiment.</p>
<p><img src="https://github.com/Way4ward17/ML---Alchohol-Effect-In-Student-Prediction/blob/master/Picture4.png" alt="" width="700" height="450" /></p>
<p><strong>4.3.3&nbsp;&nbsp;&nbsp; CLUSTERING EXPLORER</strong></p>
<p>The clustering explorer screen allow us to perform clustering experiment. We would be able to train and test our dataset in this section. The USE TRAINING SET radio button allow us to train a model if it is clicked and the SUPPLIED TEST SET allow us to test a model once a model has been trained and saved.</p>
<p>&nbsp;<img src="https://github.com/Way4ward17/ML---Alchohol-Effect-In-Student-Prediction/blob/master/Picture5.png" alt="" width="700" height="500" /></p>
<p>The first thing to do in this section is to select the algorithm to use. In my case, I would be using the K-Means algorithm</p>
<p><img src="https://github.com/Way4ward17/ML---Alchohol-Effect-In-Student-Prediction/blob/master/Picture6.png" alt="" width="700" height="450" /></p>
<p>After we have selected the K-Means algorithm, the next step is to select the number of clusters we want. In my case, I used 8 clusters, for proper categorization of the data.</p>
<p><strong>== Run information ===</strong></p>
<p>Scheme:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; weka.clusterers.SimpleKMeans -init 0 -max-candidates 100 -periodic-pruning 10000 -min-density 2.0 -t1 -1.25 -t2 -1.0 -N 8 -A "weka.core.EuclideanDistance -R first-last" -I 500 -num-slots 1 -S 10</p>
<p>Relation:&nbsp;&nbsp;&nbsp;&nbsp; datasetHardcopy-weka.filters.unsupervised.attribute.Remove-R1</p>
<p>Instances:&nbsp;&nbsp;&nbsp; 200</p>
<p>Attributes:&nbsp;&nbsp; 10</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; gender</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;morning</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; afternoon</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; night</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; examperiod</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Ist gpa</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2nd gpa</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3rd gpa</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; total</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; average</p>
<p>Test mode:&nbsp;&nbsp;&nbsp; evaluate on training data</p>
<p>=== Clustering model (full training set) ===</p>
<p>kMeans</p>
<p><img src="https://github.com/Way4ward17/ML---Alchohol-Effect-In-Student-Prediction/blob/master/Picture7.png" alt="" width="700" height="450" /></p>
<p>Time taken to build model (full training data) : 0.01 seconds</p>
<p>=== Model and evaluation on training set ===</p>
<p>&nbsp;</p>
<p>&nbsp;</p>
<p>Clustered Instances</p>
<p>0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 28 ( 14%)</p>
<p>1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 19 ( 10%)</p>
<p>2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 31 ( 16%)</p>
<p>3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 27 ( 14%)</p>
<p>4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 25 ( 13%)</p>
<p>5&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 22 ( 11%)</p>
<p>6&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 18 (&nbsp; 9%)</p>
<p>7&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 30 ( 15%)</p>
<p>&nbsp;</p>
<p><strong>4.3.4&nbsp;&nbsp;&nbsp; TESTING MODEL</strong></p>
<p>In other to test the model trained. We have to import an already converted CSV file into and arff format in the preprocessing stage. We then import it to dataset. As seen in the screenshot below, the sample dataset contain 166 samples and the test information are detailed below.</p>
<p><strong>=== Run information ===</strong></p>
<p>&nbsp;<img src="https://github.com/Way4ward17/ML---Alchohol-Effect-In-Student-Prediction/blob/master/Picture8.png" alt="" width="700" height="450" /></p>
<p>Scheme:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; weka.clusterers.SimpleKMeans -init 0 -max-candidates 100 -periodic-pruning 10000 -min-density 2.0 -t1 -1.25 -t2 -1.0 -N 8 -A "weka.core.EuclideanDistance -R first-last" -I 500 -num-slots 1 -S 10</p>
<p>Relation:&nbsp;&nbsp;&nbsp;&nbsp; datasetHardcopy copySmail-weka.filters.unsupervised.attribute.Remove-R1</p>
<p>Instances:&nbsp;&nbsp;&nbsp; 166</p>
<p>Attributes:&nbsp;&nbsp; 10</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Gender</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Morning</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Afternoon</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Night</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Examperiod</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Ist Gpa</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 2nd Gpa</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 3rd Gpa</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Total</p>
<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Average</p>
<p>Test mode:&nbsp;&nbsp;&nbsp; user supplied test set: 166 instances</p>
<p>=== Clustering model (full training set) ===</p>
<p>kMeans</p>
<p><img src="https://github.com/Way4ward17/ML---Alchohol-Effect-In-Student-Prediction/blob/master/Picture9.png" alt="" width="700" height="450" /></p>
<p>Time taken to build model (full training data) : 0 seconds</p>
<p><strong>=== Evaluation on test set ===</strong></p>
<p>Clustered Instances</p>
<p>0&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 16 ( 10%)</p>
<p>1&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 34 ( 20%)</p>
<p>2&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 25 ( 15%)</p>
<p>3&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 22 ( 13%)</p>
<p>4&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 21 ( 13%)</p>
<p>5&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 16 ( 10%)</p>
<p>6&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 22 ( 13%)</p>
<p>7&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 10 (&nbsp; 6%)</p>
<p>&nbsp;</p>
<p>Because clustering is a supervised learning. The data scientist would then scan throught the dataset to understand the similarities among the data and the clusters. From the test above. It is seen that student that doesn&rsquo;t take alcohol in the morning and little in the afternoon and doesn&rsquo;t take at night and takes little during exam time perform better and the list score are for student that takes alcohol almost everyday, cluster 5 and 1 respectively.</p>
