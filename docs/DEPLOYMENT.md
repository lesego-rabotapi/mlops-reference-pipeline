# Deployment Strategy

## Current Strategy

Direct AWS deployment is out of scope because AWS account access is unavailable.
The project will use a local-first deployment strategy:

1. Run the full pipeline locally.
2. Persist validation, feature, training, and evaluation artifacts.
3. Serve predictions through FastAPI.
4. Package the inference service with Docker.
5. Document how the same design could move to cloud infrastructure later.

This is enough to demonstrate deployment readiness without requiring live cloud
resources.

## Local Runtime Target

The primary deployment target is a local FastAPI service.

Planned endpoints:

- `/health` for readiness checks
- `/predict` for model inference
- `/metrics` for basic operational metrics

The inference service should load:

- The trained model artifact
- The fitted preprocessing artifact
- Feature metadata needed to keep inference consistent with training

## Docker Target

Docker should become the portable runtime boundary for the project.

The Docker image should:

- Install dependencies from the project dependency file
- Copy source code into the image
- Load model and preprocessing artifacts from a mounted or copied artifact path
- Start the FastAPI service with Uvicorn

No cloud credentials should be required to build or run the container.

## CI/CD Target

The CI workflow should validate the project without deploying to cloud
infrastructure.

Required CI steps:

- Create Python environment
- Install dependencies
- Run environment check
- Run validation tests
- Run feature engineering tests
- Run future training and inference tests

Deployment jobs should remain local/container-focused unless cloud access is
restored.

## Future Cloud Mapping

If cloud access becomes available later, this project can be adapted to AWS or
another cloud provider. The current local-first architecture intentionally keeps
the boundaries portable.

| Current Component | Future Cloud Deployment Option |
| --- | --- |
| Local artifact files | Object storage |
| FastAPI application | Container, VM, or serverless service |
| Docker image | Container registry and runtime |
| Local logs | Managed logging |
| Local metrics endpoint | Managed monitoring |
| Makefile / scripts | CI/CD workflow steps |

## Production Considerations

Before any real deployment, the project would need:

- Secrets management
- Authentication and authorization
- Artifact versioning policy
- Rollback procedure
- Monitoring alerts
- Resource cost controls
- Data retention policy
