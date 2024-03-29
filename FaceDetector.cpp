#include "FaceDetector.h"
#include "IO/CObjectDetectionIO.h"

CFaceDetector::CFaceDetector() : COcvDnnProcess(), CObjectDetectionTask()
{
    init();
    m_pParam = std::make_shared<CFaceDetectorParam>();
}

CFaceDetector::CFaceDetector(const std::string &name, const std::shared_ptr<CFaceDetectorParam> &pParam)
    : COcvDnnProcess(), CObjectDetectionTask(name)
{
    init();
    m_pParam = std::make_shared<CFaceDetectorParam>(*pParam);
}

void CFaceDetector::init()
{
    m_classNames = {"Unknown", "Human"};

    // Generate random colors
    std::srand(9);
    double factor = 255 / (double)RAND_MAX;

    for (size_t i=0; i<m_classNames.size(); ++i)
    {
        CColor color = {
            (uchar)((double)std::rand() * factor),
            (uchar)((double)std::rand() * factor),
            (uchar)((double)std::rand() * factor)
        };
        m_classColors.push_back(color);
    }
}

size_t CFaceDetector::getProgressSteps()
{
    return 3;
}

int CFaceDetector::getNetworkInputSize() const
{
    int size = 300;

    // Trick to overcome OpenCV issue around CUDA context and multithreading
    // https://github.com/opencv/opencv/issues/20566
    auto pParam = std::dynamic_pointer_cast<CFaceDetectorParam>(m_pParam);
    if(pParam->m_backend == cv::dnn::DNN_BACKEND_CUDA && m_bNewInput)
        size = size + (m_sign * 32);

    return size;
}

double CFaceDetector::getNetworkInputScaleFactor() const
{
    return 1.0;
}

cv::Scalar CFaceDetector::getNetworkInputMean() const
{
    return cv::Scalar(104.0, 177.0, 123.0);
}

void CFaceDetector::run()
{
    beginTaskRun();
    auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
    auto pParam = std::dynamic_pointer_cast<CFaceDetectorParam>(m_pParam);

    if (pInput == nullptr)
        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid image input", __func__, __FILE__, __LINE__);

    if (pParam == nullptr)
        throw CException(CoreExCode::INVALID_PARAMETER, "Invalid parameters", __func__, __FILE__, __LINE__);

    if (pInput->isDataAvailable() == false)
        throw CException(CoreExCode::INVALID_PARAMETER, "Empty image", __func__, __FILE__, __LINE__);

    //Force model files path
    std::string pluginDir = Utils::Plugin::getCppPath() + "/" + Utils::File::conformName(QString::fromStdString(m_name)).toStdString();
    pParam->m_structureFile = pluginDir + "/Model/res10_300x300_ssd_iter_140000.prototxt";
    pParam->m_modelFile = pluginDir + "/Model/res10_300x300_ssd_iter_140000.caffemodel";

    if (!Utils::File::isFileExist(pParam->m_modelFile))
    {
        std::cout << "Downloading model..." << std::endl;
        std::string downloadUrl = Utils::Plugin::getModelHubUrl() + "/" + m_name + "/res10_300x300_ssd_iter_140000.caffemodel";
        download(downloadUrl, pParam->m_modelFile);
    }

    CMat imgSrc;
    CMat imgOrigin = pInput->getImage();
    std::vector<cv::Mat> netOutputs;

    //Detection networks need color image as input
    if(imgOrigin.channels() < 3)
        cv::cvtColor(imgOrigin, imgSrc, cv::COLOR_GRAY2RGB);
    else
        imgSrc = imgOrigin;

    emit m_signalHandler->doProgress();

    try
    {
        if(m_net.empty() || pParam->m_bUpdate)
        {
            m_net = readDnn(pParam);
            if(m_net.empty())
                throw CException(CoreExCode::INVALID_PARAMETER, "Failed to load network", __func__, __FILE__, __LINE__);

            pParam->m_bUpdate = false;
        }
        forward(imgSrc, netOutputs, pParam);
    }
    catch(std::exception& e)
    {
        throw CException(CoreExCode::INVALID_PARAMETER, e.what(), __func__, __FILE__, __LINE__);
    }

    endTaskRun();
    emit m_signalHandler->doProgress();
    manageOutput(netOutputs[0]);
    emit m_signalHandler->doProgress();
}

void CFaceDetector::manageOutput(cv::Mat &dnnOutput)
{
    // Network produces output blob with a shape 1x1xNx7 where N is a number of
    // detections and an every detection is a vector of values
    // [batchId, classId, confidence, left, top, right, bottom]
    auto pParam = std::dynamic_pointer_cast<CFaceDetectorParam>(m_pParam);
    auto pInput = std::dynamic_pointer_cast<CImageIO>(getInput(0));
    CMat imgSrc = pInput->getImage();

    for(int i=0; i<dnnOutput.size[2]; i++)
    {
        //Detected class
        int classIndex[4] = { 0, 0, i, 1 };
        size_t classId = (size_t)dnnOutput.at<float>(classIndex);
        //Confidence
        int confidenceIndex[4] = { 0, 0, i, 2 };
        float confidence = dnnOutput.at<float>(confidenceIndex);

        if(confidence > pParam->m_confidence)
        {
            //Bounding box
            int leftIndex[4] = { 0, 0, i, 3 };
            int topIndex[4] = { 0, 0, i, 4 };
            int rightIndex[4] = { 0, 0, i, 5 };
            int bottomIndex[4] = { 0, 0, i, 6 };
            float left = dnnOutput.at<float>(leftIndex) * imgSrc.cols;
            float top = dnnOutput.at<float>(topIndex) * imgSrc.rows;
            float right = dnnOutput.at<float>(rightIndex) * imgSrc.cols;
            float bottom = dnnOutput.at<float>(bottomIndex) * imgSrc.rows;
            float width = right - left + 1;
            float height = bottom - top + 1;
            addObject(i, classId, confidence, left, top, width, height);
        }
    }
}
