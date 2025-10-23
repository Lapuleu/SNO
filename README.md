# SNO
The Code Repository of the Sumudu Neural Operator

Listed here are the Sumudu Neural Operator implementations for 1D, 2D, and 3d differential equations. We use the LNO Data files to compare consistent data. The files labeled "ode" and "pde" are the original implementations, while the files ending in "Re" are the optimized versions used to run our tests and get data for our plots.

All SNO results are stored in the Results folder, along with the Zero-shot super-resolution outputs and graphs. The code for the models is in the Main directory and uses the same hyperparameters as LNO, with only the width and degree parameters specific to SNO.

Note for SNO3dRe: It uses the same utilities3 and Adam file as in the LNO 3d Brusselator.

# LNO Replication
This is a replication of the LNO code, available here: https://github.com/qianyingcao/Laplace-Neural-Operator/. There are minor structural changes, but the code remains largely unchanged. The weights used for LNO are the same as in the LNO paper and original code.

Note: The file paths may vary depending on your application, so ensure they are accurate. Some of the Data folders have fillers; the real data can be found here: https://drive.google.com/drive/folders/1x8EYALKl2l9lxpMVy6rfj934kno4V0qB?usp=sharing. GitHub doesn't allow files over 25 mb.

# FNO Replication
This is a replication of the FNO code found here: https://github.com/ixScience/fourier_neural_operator/. There are minor updates to the saving settings to maintain consistency with LNO and SNO. We have also changed the 4-layer structure to a 1-layer Fourier Layer to maintain consistency with the parameter and layer counts.

We use the same LNO weights for FNO, with the only difference being in the activation functions which we have opted to keep fromt the original FNO code.

Note: The file paths may vary depending on your application, so ensure they are accurate. The data file can be found here: https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-.
Although the FNO paper used specific data files, we have opted to use LNO's data files for all our operators.

Note for FNO3d: It uses the same utilities3 and Adam file as in the LNO 3d Brusselator.
