@Library(['github.com/indigo-dc/jenkins-pipeline-library@release/2.1.0']) _

def projectConfig

pipeline {
    agent any
    environment {
        CPU_TAG = "${env.BRANCH_NAME == 'master' ? 'cpu' : 'cpu-${env.BRANCH_NAME}'}"
        GPU_TAG = "${env.BRANCH_NAME == 'master' ? 'gpu' : 'gpu-${env.BRANCH_NAME}'}"
    }
    stages {
        stage('SQA baseline dynamic stages') {
            steps {
                script {
                    projectConfig = pipelineConfig()
                    buildStages(projectConfig)
                }
            }
            post {
                cleanup {
                    cleanWs()
                }
            }
        }
    }
}
