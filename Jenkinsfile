pipeline{
    agent any

    environment{
        VENV_DIR = 'venv'
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
    }

}