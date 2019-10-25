Model B

########################################################################################
|| Scaling - input || Scaling - output ||  Splitting  ||  gammas  || Cross-Validation ||
########################################################################################
||  StandardScaler ||       No         ||  80 - 20 a  ||  fixed   ||      LOO-CV      ||
||  StandardScaler ||       Yes        ||  80 - 20 a  ||  fixed   ||      LOO-CV      ||
||  StandardScaler ||       No         ||  80 - 20 a  || adaptive ||      LOO-CV      ||
||  StandardScaler ||       Yes        ||  80 - 20 a  || adaptive ||      LOO-CV      ||
||  StandardScaler ||       No         ||  80 - 20 b  ||  fixed   ||      LOO-CV      ||
||  StandardScaler ||       Yes        ||  80 - 20 b  ||  fixed   ||      LOO-CV      ||
||  StandardScaler ||       No         ||  80 - 20 b  || adaptive ||      LOO-CV      ||
||  StandardScaler ||       Yes        ||  80 - 20 b  || adaptive ||      LOO-CV      ||
||   RobustSacler  ||       No         ||  80 - 20 a  ||  fixed   ||      LOO-CV      ||
||   RobustSacler  ||       Yes        ||  80 - 20 a  ||  fixed   ||      LOO-CV      ||
||   RobustSacler  ||       No         ||  80 - 20 a  || adaptive ||      LOO-CV      ||
||   RobustSacler  ||       Yes        ||  80 - 20 a  || adaptive ||      LOO-CV      ||
||   RobustSacler  ||       No         ||  80 - 20 b  ||  fixed   ||      LOO-CV      ||
||   RobustSacler  ||       Yes        ||  80 - 20 b  ||  fixed   ||      LOO-CV      ||
||   RobustSacler  ||       No         ||  80 - 20 b  || adaptive ||      LOO-CV      ||
||   RobustSacler  ||       Yes        ||  80 - 20 b  || adaptive ||      LOO-CV      ||
