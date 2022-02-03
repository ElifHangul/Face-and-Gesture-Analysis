% Basic script for Face Detection Challenge
% --------------------------------------------------------------------
% AGC Challenge  
% Universitat Pompeu Fabra
%

% Load challenge Training data
load AGC_Challenge1_Training.mat
disp(AGC_Challenge1_TRAINING(11).faceBox);

% Provide the path to the input images, for example 
% 'C:\AGC_Challenge\images\'
imgPath = ["TRAINING\"];

% Initialize results structure
DetectionSTR = struct();

% Initialize timer accumulator
total_time = 0;

% Process all images in the Training set
for j = 1 : length( AGC_Challenge1_TRAINING )
    A = imread( sprintf('%s%s',...
        imgPath, AGC_Challenge1_TRAINING(j).imageName ));    
    
    try
        % Timer on
        tic;
        
        % ###############################################################
        % Your face detection function goes here. It must accept a single
        % input parameter (the input image A) and it must return one or
        % more bounding boxes corresponding to the facial images found 
        % in image A, specificed as [x1 y1 x2 y2]
        % Each bounding box that is detected will be indicated in a 
        % separate row in det_faces
        
        det_faces = MyFaceDetectionFunction( A );        
        % ###############################################################
        
        % Update total time
        tt = toc;
        total_time = total_time + tt;
        
    catch
        % If the face detection function fails, it will be assumed that no
        % face was detected for this input image
        det_faces = [];
    end

    % Store the detection(s) in the resulst structure
    DetectionSTR(j).det_faces = det_faces;
end
   
% Compute detection score
FD_score = CHALL_AGC_ComputeDetScores(...
    DetectionSTR, AGC_Challenge1_TRAINING, 0);

% Display summary of results
fprintf(1, '\n F1-score: %.2f%% \t Total time: %dm %ds\n', ...
    100 * FD_score, int16( total_time/60),...
    int16(mod( total_time, 60)) );


function bbox = MyFaceDetectionFunction( A )
% creating the cascadeobjectdetector for face detection
FDetect = vision.CascadeObjectDetector();
% look into 7 face bounding boxes to merge
FDetect.MergeThreshold = 7 ;
% different scale factor values are tried to come up with the optimum one
FDetect.ScaleFactor=1.1704;
% the returning bounding box is defined as [x,y,width,height]. To transform
% this format to [x1,y1,x2,y2] format we wrote the values as [x,y,x+width,y+height]
BB = step(FDetect, A);
x2 = BB(1)+BB(3);
y2 = BB(2)+BB(4);
% return the bounding box with the new parameters
bbox = [BB(1) BB(2) x2 y2];
end