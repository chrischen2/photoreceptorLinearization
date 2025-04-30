%% -------------------------------------------------------
%% Step 1: Initialize model parameters
% Parameters below are consensus parameters for each photoreceptor type.
% see https://github.com/chrischen2/coneLinearization.wiki.git for mathematical details on
% the biophysical model
%% -------------------------------------------------------
modelType = 'peripheralPrimateCone'; % Example model type
params = initPhotoreceptorParams(modelType);
params.darkCurrent = params.gdark^params.n * params.k;

%% -------------------------------------------------------
%% Step 2: Define Linear Model Coefficients to generate target responses
% These are coefficients of a linear model fit to the responses of the full
% model of each type to low contrast (i.e. linear range) gaussian noise
% stimuli. Models are fit at several discrete light levels - so that a
% given linear model is the approximation to the behavior of the
% phototransduction cascade at the specified light level.

% The output structure linearModelCoef for each photoreceptor type contains arrays with four columns:
%  - The first column represents the light level.
%  - The second column is the Scaling factor.
%  - The third column is the Rising time constant.
%  - The fourth column is the Decaying time constant.

% refer to fitLinearPRModel.m for guidance on how to compute linear model
% coefficient for more light levels of specific photoreceptor type
% see https://github.com/chrischen2/coneLinearization.wiki.git for details on
% the coefficient of Linear model 
%% -------------------------------------------------------

linearModelCoef = defineLinearModelCoefficients();
coefLin=linearModelCoef.(modelType);
coefLin=coefLin(3,2:4);   % mean light level of 5000 R*/cone/s see defineLinearModelCoefficients.m 

%% Step 3: generate initial stimulus
%% -------------------------------------------------------
% Set the mean intensity of the stimulus in R*/s (Rhodopsin activations per second).
meanIntensity = 5000;  % R*/s
% Set the number of points for smoothing the stimulus. This affects the smoothness of the generated stimulus.
params.smoothPts = 100;  % Smoothing of stimulus
% Flag to match the power of the stimulus and the estimate. 
% Set to 0 for matching target responses, indicating no power matching is applied.
params.matchPower = 0;  
% Flag to match the mean current of the stimulus.
% Set to 0, indicating the mean current is not matched to the stimulus.
params.matchMean = 0;  
% Number of points for allowing the model to settle before the main stimulus is applied.
params.prePts = 10000;  
% Total number of points in the stimulus.
numPts = 100000;  % Length of stimulus
% Calculate the mean photon flux based on the mean intensity and the time step.
MeanPhoFlux = meanIntensity * params.timeStep;
% Create a time vector for the stimulus based on the number of points and the time step.
params.tme = [1:numPts] * params.timeStep;
% Set parameters for the linear model based on previously determined coefficients.
% These parameters are used in generating the stimulus response.
params.ScFact = coefLin(1);  % Scaling factor
params.TauR = coefLin(2);    % Rising time constant
params.TauD = coefLin(3);    % Decaying time constant

%%
% Example 1: sinusoid
% High contrast sinusoid at specified frequency

sineFreq = 3; % in Hz
params.stm = (0.9 * sin(2*pi*params.tme*sineFreq) + 1) * meanIntensity; 
params.stm(1:numPts/2) = meanIntensity; % initial time to settle
params.stm = params.stm * params.timeStep;

%%
% Example 2: noise
% smoothed gaussian noise

rng(1);
params.stm = filter(gausswin(params.smoothPts), 1, normrnd(MeanPhoFlux, MeanPhoFlux*5, 1, numPts)) / sum(gausswin(params.smoothPts));
params.stm(1:numPts/2) = meanIntensity * params.timeStep;
params.stm = params.stm * params.timeStep;

%%
% Example 3: flashes and step

flashFact = 5; % brightness of flash relative to mean
stepFact = 2; % factor to change mean intensity 
flashDuration = 10;
flashTime = [numPts/2-3000 3*numPts/4-3000];

% add step
params.stm = meanIntensity * ones(size(params.tme));
params.stm(numPts/2+1:3*numPts/4) = meanIntensity * stepFact;

% add flashes
params.stm(flashTime(1)+1:flashTime(1)+flashDuration) = params.stm(flashTime(1)+1:flashTime(1)+flashDuration) + meanIntensity * flashFact;
params.stm(flashTime(2)+1:flashTime(2)+flashDuration) = params.stm(flashTime(2)+1:flashTime(2)+flashDuration) + meanIntensity * flashFact;

params.stm = params.stm * params.timeStep;

%%
% Example 4: response speeding

timeFact = 0.5;

% parameters for linear model
params.ScFact = coefLin(1);
params.TauR = coefLin(2)/timeFact; 
params.TauD = coefLin(3)/timeFact; 
params.TauP = coefLin(4)/timeFact; 
params.Phi = coefLin(5);

flashFact = 20; % brightness of flash relative to mean
flashDuration = 10;

params.stm = meanIntensity * ones(size(params.tme));
params.stm(numPts/2+1:numPts/2+flashDuration) = meanIntensity*flashFact;

params.stm = params.stm * params.timeStep;

%% -------------------------------------------------------
%% Step 4: generate target response and new stimulus
%% -------------------------------------------------------

% Full response to the original stimulus as a reference
% This step involves running the full biophysical model with the original stimulus.
params.biophysFlag = 1;  % Flag set to use the full biophysical model
params = BiophysModel(params);  % Run the biophysical model
fullResponse = params.response;  % Store the full response for later comparison

% Copy the original stimulus for later use
originalStm = params.stm;

% Linear response to the original stimulus
% Here, the linearized version of the model is used to generate a response.
params.biophysFlag = 0;  % Flag set to use the linearized model
params = BiophysModel(params);  % Run the linearized model
% Adjust the linear response to align with the full response at a specific point (numPts/4)
linearResponse = params.response - params.response(numPts/4) + fullResponse(numPts/4);

% Use model inversion to generate a stimulus that causes the full model output to
% match the target (linear) response.
params.CombinedStim = params.stm;  % Store the original stimulus
params.CombinedResponse = linearResponse;  % Use the linear response as the target
results = EstimateStmFromPhotocurrent(params);  % Estimate the stimulus to match the target response

% Plot the original and modified stimuli and the corresponding responses
params.stm = results.rawEstimate;  % Modified stimulus estimated to match the linear response
params.biophysFlag = 1;  % Use the full biophysical model
params = BiophysModel(params);  % Run the model with the modified stimulus

figure; clf;  % Create a new figure and clear any existing plots
% Plot the original and modified stimuli
subplot(1, 2, 1);
plot(params.tme, originalStm/params.timeStep, 'LineWidth', 2); hold on;  % Original stimulus plot
plot(params.tme, results.estimate, 'LineWidth', 2);  % Modified stimulus plot
% ylim([0 meanIntensity*params.timeStep*8]);  
legend('original', 'modified');  
title('Stimuli');  
xlabel('sec');  
ylabel('R*/s');  

% Plot the full and linear responses
subplot(1, 2, 2);
plot(params.tme, fullResponse, 'LineWidth', 2); hold on;  % Full response plot
plot(params.tme, linearResponse, 'LineWidth', 2);  % Linear response plot
legend('full', 'linear');  
title('Responses');  
xlabel('sec'); 
ylabel('pA');  

params.stm=originalStm;
