pipeline{
    agent any

    environment{
        VENV_DIR = 'venv'
        GCP_PROJECT = 'mlopsproject1-461016'
        GCLOUD_PATH = '/var/jenkins_home/google-cloud-sdk/bin'
    }

    stages{
        stage("Cloning the Github Repository to Jenkins"){
            steps{
                script{
                    echo 'Cloning the Github repository to Jenkins...'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/RajaSekhar899/MLOps_Udemy_Project_1.git']])
                }
            }
        }
        stage("Setting up Virtual Environment and Installing the dependencies"){
            steps{
                script{
                    echo 'Setting up Virtual Environment and Installing the dependencies...'
                    sh '''
                    # Create a virtual environment
                    python -m venv ${VENV_DIR}
                    # Activate the virtual environment and install dependencies
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip
                    pip install -e .
                    '''
                }
            }
        }

        stage("Building and pushing docker image to GCR"){
            steps{
                withCredentials([file(credentialsId: 'gcp-key', variable: 'GOOGLE_APPLICATION_CREDENTIALS')]){
                    script{
                        echo 'Building and pushing docker image to GCR...'
                        sh '''
                        export PATH=$PATH:${GCLOUD_PATH}

                        # Activate and authenticate the service account
                        gcloud auth activate-service-account --key-file=${GOOGLE_APPLICATION_CREDENTIALS}

                        # Set the project
                        gcloud config set project ${GCP_PROJECT}

                        # Configure Docker with GCR
                        gcloud auth configure-docker --quiet

                        # Build the Docker image
                        docker build -t gcr.io/${GCP_PROJECT}/mlops-project-1:latest .

                        # Push the Docker image to GCR
                        docker push gcr.io/${GCP_PROJECT}/mlops-project-1:latest
                                                
                        '''
                    }
                }
            }
        }
    }

}