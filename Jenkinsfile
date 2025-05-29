pipeline{
    agent any

    stages{
        stage("Cloning the Github Repository to Jenkins"){
            steps{
                script{
                    echo 'Cloning the Github repository to Jenkins...'
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/RajaSekhar899/MLOps_Udemy_Project_1.git']])
                }
            }
        }
    }

}