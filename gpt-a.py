# Retry with fixes: access fitted TF-IDF from pipeline's preprocessor before extracting feature names.
from sklearn.exceptions import NotFittedError

# ensure pipeline is fitted (it was fitted earlier)
try:
    fitted_tfidf = pipe.named_steps["pre"].named_transformers_["tfidf"]
    feature_names = fitted_tfidf.get_feature_names_out()
    coef = pipe.named_steps["clf"].coef_.ravel()
    tfidf_coef = coef[:len(feature_names)]
    top_pos_idx = np.argsort(tfidf_coef)[-10:][::-1]
    top_neg_idx = np.argsort(tfidf_coef)[:10]
    top_pos = [(feature_names[i], tfidf_coef[i]) for i in top_pos_idx]
    top_neg = [(feature_names[i], tfidf_coef[i]) for i in top_neg_idx]

    print("Baseline (uncalibrated) ROC AUC:", roc_auc_uncal)
    print("Baseline (uncalibrated) Brier:", brier_uncal)
    print("Calibrated ROC AUC:", roc_auc_cal)
    print("Calibrated Brier:", brier_cal)
    print("\nTop TF-IDF features associated with HUMAN label (positive coeffs):")
    for f,c in top_pos:
        print(f"{f}: {c:.4f}")
    print("\nTop TF-IDF features associated with AI label (negative coeffs):")
    for f,c in top_neg:
        print(f"{f}: {c:.4f}")

    # Show a small table with predicted probabilities for a few test samples
    test_samples = X_test.reset_index(drop=True).copy()
    test_samples["true_label"] = y_test.reset_index(drop=True)
    test_samples["proba_uncal"] = y_proba_uncal
    test_samples["proba_cal"] = y_proba_cal
    display(test_samples.sample(8, random_state=2))
except NotFittedError as e:
    print("Pipeline not fitted:", e)
