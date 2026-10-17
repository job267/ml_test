"""One test scoring pass, two predeclared masks; never used for model selection."""
import torch


def validation_mask_rows(users, validation, legacy):
    if legacy:
        # Exact author getUserValidItems: silently omits users absent from validation.
        return [validation[u] for u in users if u in validation]
    return [validation.get(u, []) for u in users]


@torch.no_grad()
def evaluate_public_test(dataset, model, predictor, world, procedure):
    model.eval()
    item_pop = torch.tensor([dataset.item_inter[i] for i in range(dataset.m_item)], dtype=torch.float32, device=world.device)
    stage = torch.full((dataset.m_item,), world.config["period"] + 1, device=world.device, dtype=torch.long)
    users = list(dataset.test_dict)
    truth, rankings = [], {"author_original_mask": [], "corrected_user_aligned_mask": []}
    batch_size = world.config["test_u_batch_size"]
    for start in range(0, len(users), batch_size):
        batch = users[start:start + batch_size]
        ids = torch.tensor(batch, device=world.device, dtype=torch.long)
        scores = model.getUsersRating(ids, predictor, world, stage, item_pop)
        if not torch.isfinite(scores).all():
            raise FloatingPointError("Non-finite public test scores")
        for row, pos in enumerate(dataset.getUserPosItems(batch)):
            scores[row, pos] = -(1 << 10)  # Preserve author masking sentinel.
        for name, legacy in (("author_original_mask", True), ("corrected_user_aligned_mask", False)):
            masked = scores.clone()
            for row, known in enumerate(validation_mask_rows(batch, dataset.valid_dict, legacy)):
                masked[row, known] = -(1 << 10)
            rankings[name].extend(torch.topk(masked, max(world.topks), dim=1).indices.cpu())
        truth.extend(dataset.test_dict[u] for u in batch)
    result = {}
    for name, predicted in rankings.items():
        p, r, n, m = procedure.computeTopNAccuracy(truth, predicted, [10, 20, 50, 100])
        result[name] = {"precision": p, "recall": r, "ndcg": n, "mrr": m}
    return result
