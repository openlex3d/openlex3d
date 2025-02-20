def evaluate_rank(matches, iou_threshold):
    """
    Evaluate the rank metric using predictions.

    Args:
        matches (dict): Dictionary with structure:
            {scene_id: {"gt": {"object": [gt_instances]},
                        "pred": {"object": [pred_instances]}}}
        iou_threshold (float): IoU threshold for considering a prediction as a valid match.

    Returns:
        avg_inverse_rank (float): The average inverse rank.
        scene_query_ranks (dict): Dictionary with per-scene, per-query adjusted ranks.
            Format: {scene_id: {query_id: [list_of_adjusted_ranks]}}
    """
    scene_query_ranks = {}

    # Process each scene individually.
    for scene_id, match in matches.items():
        pred_instances = match["pred"]["object"]
        gt_instances = match["gt"]["object"]

        # Create a dictionary to track which predictions have already been used.
        pred_visited = {pred["uuid"]: False for pred in pred_instances}

        # For each ground-truth instance, find the best matching prediction (if any)
        query_ranks = {}  # Stores, per query, a list of raw ranks (one per GT instance)
        for gt in gt_instances:
            qid = gt["query_id"]
            query_ranks.setdefault(qid, [])

            best_rank = None
            best_iou = 0.0
            best_pred_uuid = None

            # Iterate over all predictions that were matched to this GT instance.
            for pred in gt.get("matched_pred", []):
                uuid = pred["uuid"]

                # Skip predictions already used for another GT.
                if pred_visited.get(uuid, False):
                    continue

                # Compute the IoU between the GT and prediction.
                intersection = pred.get("intersection", 0)
                union = gt["vert_count"] + pred["vert_count"] - intersection
                iou = intersection / union if union > 0 else 0.0

                # If the IoU is above the threshold and is the best seen so far, select this prediction.
                if iou > iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_rank = pred["rank"]
                    best_pred_uuid = uuid

            # Record the rank if a valid prediction was found; otherwise, record -1.
            if best_rank is None:
                query_ranks[qid].append(-1)
            else:
                query_ranks[qid].append(best_rank)
                pred_visited[best_pred_uuid] = True  # Mark the prediction as used.

        # Adjust the rank list for each query:
        # For each query, sort the valid ranks (ignoring -1), then subtract the positional index
        # from each valid rank. (Leave any -1 entries unchanged.)
        adjusted_query_ranks = {}
        for qid, ranks in query_ranks.items():
            valid_ranks = sorted([r for r in ranks if r != -1])
            invalid_ranks = [r for r in ranks if r == -1]
            adjusted_valid_ranks = []
            for i, r in enumerate(valid_ranks):
                adjusted_valid_ranks.append(r - i)  # subtract the positional index

            # Combine and sort: the -1 values remain unchanged.
            adjusted_ranks = sorted(
                adjusted_valid_ranks + invalid_ranks, key=lambda x: (x == -1, x)
            )
            adjusted_query_ranks[qid] = adjusted_ranks

        # Save the per-query adjusted ranks for the current scene.
        scene_query_ranks[scene_id] = adjusted_query_ranks

    # Concatenate the adjusted ranks from all queries and scenes.
    final_rank_list = []
    for scene_data in scene_query_ranks.values():
        for rank_list in scene_data.values():
            final_rank_list.extend(rank_list)
    # Sort the final list (ensuring that -1 values remain unchanged).
    final_rank_list = sorted(final_rank_list, key=lambda x: (x == -1, x))

    # # print query_ids with only -1 ranks
    # for scene_id, query_data in scene_query_ranks.items():
    #     for query_id, ranks in query_data.items():
    #         if all(r == -1 for r in ranks):
    #             print(f"Scene {scene_id}, Query {query_id}: All ranks are -1.")

    # Compute the average inverse rank:
    # For valid ranks (r > 0), use 1/r; for -1 or r <= 0, use 0.
    inverse_ranks = [0.0 if r == -1 or r <= 0 else 1.0 / r for r in final_rank_list]
    avg_inverse_rank = sum(inverse_ranks) / len(inverse_ranks) if inverse_ranks else 0.0

    return avg_inverse_rank, scene_query_ranks
