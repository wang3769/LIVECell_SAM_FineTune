pipeline {
  agent any

  environment {
    ACR_NAME   = "semmlacr123"
    ACR_LOGIN  = "semmlacr123.azurecr.io"
    IMAGE_REPO = "sem-sam-api"
    IMAGE_TAG  = "${env.BUILD_NUMBER}"         // simple tag
    FULL_IMAGE = "${ACR_LOGIN}/${IMAGE_REPO}:${IMAGE_TAG}"

    INFER_HOST = "20.83.126.186"                 // <-- put sem-infer-vm public IP
    INFER_PORT = "22"
    CONTAINER_NAME = "sem-sam-api"
  }

  stages {
    stage('Checkout') {
      steps {
        checkout scm
      }
    }

    stage('Build Docker image') {
      steps {
        sh """
          docker build -t ${FULL_IMAGE} .
        """
      }
    }

    stage('Login to ACR') {
      steps {
        withCredentials([usernamePassword(credentialsId: 'acr-admin',
                                          usernameVariable: 'ACR_USER',
                                          passwordVariable: 'ACR_PASS')]) {
          sh """
            echo "$ACR_PASS" | docker login ${ACR_LOGIN} -u "$ACR_USER" --password-stdin
          """
        }
      }
    }

    stage('Push to ACR') {
      steps {
        sh """
          docker push ${FULL_IMAGE}
          docker tag ${FULL_IMAGE} ${ACR_LOGIN}/${IMAGE_REPO}:latest
          docker push ${ACR_LOGIN}/${IMAGE_REPO}:latest
        """
      }
    }

    stage('Deploy to inference VM') {
        steps {
            withCredentials([usernamePassword(credentialsId: 'acr-admin',
                                            usernameVariable: 'ACR_USER',
                                            passwordVariable: 'ACR_PASS')]) {
            sshagent(credentials: ['infer-vm-ssh']) {
                sh """
                ssh -o StrictHostKeyChecking=no -p ${INFER_PORT} azureuser@${INFER_HOST} '
                    set -e
                    echo "${ACR_PASS}" | docker login ${ACR_LOGIN} -u "${ACR_USER}" --password-stdin
                    docker pull ${ACR_LOGIN}/${IMAGE_REPO}:latest

                    docker rm -f ${CONTAINER_NAME} || true
                    docker run -d --restart unless-stopped \\
                    --name ${CONTAINER_NAME} \\
                    -p 8000:8000 \\
                    ${ACR_LOGIN}/${IMAGE_REPO}:latest
                '
                """
            }
            }
        }
    }

  }
    
}
