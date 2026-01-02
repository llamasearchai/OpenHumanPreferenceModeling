# AWS Infrastructure Definition for OpenHumanPreferenceModeling

provider "aws" {
  region = "us-east-1"
}

# VPC
resource "aws_vpc" "main" {
  cidr_block = "10.0.0.0/16"
  enable_dns_hostnames = true
  
  tags = {
    Name = "ohpm-vpc"
  }
}

# EKS Cluster
resource "aws_eks_cluster" "main" {
  name     = "ohpm-prod-cluster"
  role_arn = aws_iam_role.eks_cluster_role.arn

  vpc_config {
    subnet_ids = [aws_subnet.private_subnet.id]
  }
}

# Node Groups
resource "aws_eks_node_group" "cpu_nodes" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "cpu-nodes"
  node_role_arn   = aws_iam_role.eks_node_role.arn
  subnet_ids      = [aws_subnet.private_subnet.id]
  instance_types  = ["c5.4xlarge"]

  scaling_config {
    desired_size = 2
    max_size     = 10
    min_size     = 1
  }
}

resource "aws_eks_node_group" "gpu_nodes" {
  cluster_name    = aws_eks_cluster.main.name
  node_group_name = "gpu-nodes"
  node_role_arn   = aws_iam_role.eks_node_role.arn
  subnet_ids      = [aws_subnet.private_subnet.id]
  instance_types  = ["p3.8xlarge"]
  
  scaling_config {
    desired_size = 1
    max_size     = 5
    min_size     = 0
  }
}

# RDS
resource "aws_db_instance" "default" {
  allocated_storage    = 100
  engine               = "postgres"
  engine_version       = "13.4"
  instance_class       = "db.r5.xlarge"
  name                 = "ohpm_db"
  username             = "admin"
  password             = var.db_password
  parameter_group_name = "default.postgres13"
  skip_final_snapshot  = true
}

# S3 Buckets
resource "aws_s3_bucket" "model_artifacts" {
  bucket = "ohpm-model-artifacts"
  acl    = "private"
}

resource "aws_s3_bucket" "training_data" {
  bucket = "ohpm-training-data"
  acl    = "private"
}
