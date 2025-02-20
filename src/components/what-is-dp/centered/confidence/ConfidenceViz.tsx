import { ConfidenceChart } from "./ConfidenceChart";

export function ConfidenceViz() {
  return (
    <>
      <div className="prose prose-lg max-w-3xl mx-auto mb-12 text-center">
        <p className="text-xl">
          Here, we&apos;ve trained a logistic regression model privately and
          non-privately. Plotted below is the model&apos;s confidence in its
          prediction for either true or false. Try to identify which datum was
          used in the training set!
        </p>
        <ConfidenceChart />
      </div>
    </>
  );
}
