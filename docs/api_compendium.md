# API Compendium

## Annotation Interface
### `GET /api/tasks/next`
- **Params**: `user_id`
- **Returns**: `TASK_PAYLOAD`
- **Desc**: Next annotation task from Active Learner.

### `POST /api/annotations`
- **Body**: `{ task_id, choice, ... }`
- **Desc**: Submit user feedback.

## DPO Model
### `POST /api/predict`
- **Body**: `{ state_vector }`
- **Returns**: `PROB_DIST`
- **Desc**: Predict next best action.

## Monitoring
### `GET /api/metrics`
- **Desc**: Prometheus scrape target / JSON metrics.

### `GET /api/alerts`
- **Desc**: Active alerts list.

## Calibration
### `POST /api/calibration/recalibrate`
- **Body**: `{ validation_data_uri, target_ece, max_iterations }`
- **Desc**: Trigger temperature scaling recalibration.

### `POST /api/calibration/predictions`
- **Body**: `{ confidence, correct }`
- **Desc**: Record sampled prediction outcomes for calibration monitoring.

## Privacy
### `POST /api/gradients/upload`
- **Body**: `{ encrypted_grads }`
- **Desc**: Federated learning update.
