import { ConfidenceChart } from "./ConfidenceChart";

export function ConfidenceViz() {
  return (
    <>
      <div className="prose prose-lg max-w-3xl mx-auto mb-12 text-center">
        <p className="text-xl">
          Here, we&apos;ve trained two logistic regression models, one privately
          and one non-privately, on the same task: predicting whether an image
          is a hotdog or not. We trained these models on the same training data
          and both achieve similar test accuracies. Plotted below is the
          model&apos;s confidence that the image provided is a hotdog. Try to
          identify which image was used in the training set!{" "}
          <span className="relative group inline-block">
            <span className="underline decoration-dotted cursor-help">
              (Hint)
            </span>
            <span className="absolute top-full left-1/2 transform -translate-x-1/2 mt-2 w-64 p-3 bg-primary-gray text-primary-white rounded-lg text-sm opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-10">
              Does either model predict an image with a weirdly high confidence?
              Also, notice that all the privately released predictions
              probabilities seem relatively unconfident, but the test accuracy
              shows promising generalizability!
            </span>
          </span>{" "}
          <span className="relative group inline-block">
            <span className="underline decoration-dotted cursor-help">
              (Answer)
            </span>
            <span className="absolute top-full left-1/2 transform -translate-x-1/2 mt-2 w-64 p-3 bg-primary-gray text-primary-white rounded-lg text-sm opacity-0 invisible group-hover:opacity-100 group-hover:visible transition-all duration-200 z-10">
              Image 2 was in the training set! Notice in the non-private model
              how image 2 is exceptionally condfident. An attacker could
              identify this datum as part of the training set (or incorrectly
              assume so) and proceed to harm that individual.
            </span>
          </span>
        </p>
        <ConfidenceChart />
      </div>
    </>
  );
}
