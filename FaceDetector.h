#ifndef FACEDETECTOR_H
#define FACEDETECTOR_H

#include "FaceDetectorGlobal.h"
#include "Process/OpenCV/dnn/COcvDnnProcess.h"
#include "Widget/OpenCV/dnn/COcvWidgetDnnCore.h"
#include "CPluginProcessInterface.hpp"

//------------------------------//
//----- CFaceDetectorParam -----//
//------------------------------//
class FACEDETECTORSHARED_EXPORT CFaceDetectorParam: public COcvDnnProcessParam
{
    public:

        CFaceDetectorParam() : COcvDnnProcessParam()
        {
            m_framework = Framework::CAFFE;
        }

        void        setParamMap(const UMapString& paramMap) override
        {
            COcvDnnProcessParam::setParamMap(paramMap);
            m_confidence = std::stod(paramMap.at("confidence"));
            m_nmsThreshold = std::stod(paramMap.at("nmsThreshold"));
        }

        UMapString  getParamMap() const override
        {
            auto paramMap = COcvDnnProcessParam::getParamMap();
            paramMap.insert(std::make_pair("confidence", std::to_string(m_confidence)));
            paramMap.insert(std::make_pair("nmsThreshold", std::to_string(m_nmsThreshold)));
            return paramMap;
        }

    public:

        double m_confidence = 0.5;
        double m_nmsThreshold = 0.4;
};

//-------------------------//
//----- CFaceDetector -----//
//-------------------------//
class FACEDETECTORSHARED_EXPORT CFaceDetector: public COcvDnnProcess
{
    public:

        CFaceDetector();
        CFaceDetector(const std::string& name, const std::shared_ptr<CFaceDetectorParam>& pParam);

        size_t      getProgressSteps() override;
        int         getNetworkInputSize() const override;
        double      getNetworkInputScaleFactor() const override;
        cv::Scalar  getNetworkInputMean() const override;

        void        run() override;

    private:

        void        manageOutput(cv::Mat &dnnOutput);
};

//--------------------------------//
//----- CFaceDetectorFactory -----//
//--------------------------------//
class FACEDETECTORSHARED_EXPORT CFaceDetectorFactory : public CTaskFactory
{
    public:

        CFaceDetectorFactory()
        {
            m_info.m_name = "infer_face_detector";
            m_info.m_shortDescription = QObject::tr("Deep learning based face detector").toStdString();
            m_info.m_description = QObject::tr("This model was included in OpenCV from version 3.3. "
                                               "It is based on Single-Shot-Multibox detector and uses ResNet-10 Architecture as backbone. "
                                               "The model was trained using images available from the web, but the source is not disclosed.").toStdString();
            m_info.m_docLink = "https://www.pyimagesearch.com/2018/02/26/face-detection-with-opencv-and-deep-learning/";
            m_info.m_path = QObject::tr("Plugins/C++/Face").toStdString();
            m_info.m_iconPath = "Icon/icon.png";
            m_info.m_authors = "OpenCV, Ikomia";
            m_info.m_license = "3-clause BSD License";
            m_info.m_repo = "https://github.com/opencv/opencv/tree/master/samples/dnn";
            m_info.m_version = "1.2.0";
            m_info.m_keywords = "deep,learning,detection,caffe," + Utils::Plugin::getArchitectureKeywords();
        }

        virtual WorkflowTaskPtr create(const WorkflowTaskParamPtr& pParam) override
        {
            auto paramPtr = std::dynamic_pointer_cast<CFaceDetectorParam>(pParam);
            if(paramPtr != nullptr)
                return std::make_shared<CFaceDetector>(m_info.m_name, paramPtr);
            else
                return create();
        }
        virtual WorkflowTaskPtr create() override
        {
            auto paramPtr = std::make_shared<CFaceDetectorParam>();
            assert(paramPtr != nullptr);
            return std::make_shared<CFaceDetector>(m_info.m_name, paramPtr);
        }
};

//-------------------------------//
//----- CFaceDetectorWidget -----//
//-------------------------------//
class FACEDETECTORSHARED_EXPORT CFaceDetectorWidget: public COcvWidgetDnnCore
{
    public:

        CFaceDetectorWidget(QWidget *parent = Q_NULLPTR): COcvWidgetDnnCore(parent)
        {
            init();
        }
        CFaceDetectorWidget(WorkflowTaskParamPtr pParam, QWidget *parent = Q_NULLPTR): COcvWidgetDnnCore(pParam, parent)
        {
            m_pParam = std::dynamic_pointer_cast<CFaceDetectorParam>(pParam);
            init();
        }

    private:

        void init()
        {
            if(m_pParam == nullptr)
                m_pParam = std::make_shared<CFaceDetectorParam>();

            auto pParam = std::dynamic_pointer_cast<CFaceDetectorParam>(m_pParam);
            assert(pParam);

            auto pSpinConfidence = addDoubleSpin(tr("Confidence"), pParam->m_confidence, 0.0, 1.0, 0.1, 2);
            auto pSpinNmsThreshold = addDoubleSpin(tr("NMS threshold"), pParam->m_nmsThreshold, 0.0, 1.0, 0.1, 2);

            //Connections
            connect(pSpinConfidence, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [&](double val)
            {
                auto pParam = std::dynamic_pointer_cast<CFaceDetectorParam>(m_pParam);
                assert(pParam);
                pParam->m_confidence = val;
            });
            connect(pSpinNmsThreshold, QOverload<double>::of(&QDoubleSpinBox::valueChanged), [&](double val)
            {
                auto pParam = std::dynamic_pointer_cast<CFaceDetectorParam>(m_pParam);
                assert(pParam);
                pParam->m_nmsThreshold = val;
            });
        }

        void onApply() override
        {
            emit doApplyProcess(m_pParam);
        }
};

//--------------------------------------//
//----- CFaceDetectorWidgetFactory -----//
//--------------------------------------//
class FACEDETECTORSHARED_EXPORT CFaceDetectorWidgetFactory : public CWidgetFactory
{
    public:

        CFaceDetectorWidgetFactory()
        {
            m_name = "infer_face_detector";
        }

        virtual WorkflowTaskWidgetPtr   create(WorkflowTaskParamPtr pParam)
        {
            return std::make_shared<CFaceDetectorWidget>(pParam);
        }
};

//-----------------------------------//
//----- Global plugin interface -----//
//-----------------------------------//
class FACEDETECTORSHARED_EXPORT CFaceDetectorInterface : public QObject, public CPluginProcessInterface
{
    Q_OBJECT
    Q_PLUGIN_METADATA(IID "ikomia.plugin.process")
    Q_INTERFACES(CPluginProcessInterface)

    public:

        virtual std::shared_ptr<CTaskFactory> getProcessFactory()
        {
            return std::make_shared<CFaceDetectorFactory>();
        }

        virtual std::shared_ptr<CWidgetFactory> getWidgetFactory()
        {
            return std::make_shared<CFaceDetectorWidgetFactory>();
        }
};

#endif // FACEDETECTOR_H
