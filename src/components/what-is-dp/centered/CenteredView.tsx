import Link from "next/link";
import Image from "next/image";
import { HistogramViz } from "./histogram/HistogramViz";
import { ConfidenceViz } from "./confidence/ConfidenceViz";
import { useState } from "react";
import { getPublicPath } from "@/lib/utils";

export function CenteredView() {
  const [showRevealDetails, setShowRevealDetails] = useState(false);

  return (
    <div className="bg-primary-white relative z-10 pt-4">
      {/* Sixth Paragraph - Full Width Centered */}
      <section className="min-h-screen flex flex-col justify-center items-center p-6 md:p-12">
        <div className="prose prose-lg max-w-3xl mx-auto text-center">
          <p className="text-base md:text-xl">
            What&apos;s important to note is that Differential Privacy is a
            property of an algorithm, not a property of the data. Another
            intuitive{" "}
            <Link
              href="https://en.wikipedia.org/wiki/Privacy-enhancing_technologies"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-accent cursor-pointer transition-colors underline decoration-dotted"
            >
              Privacy-Enhancing Technology
            </Link>{" "}
            might be to anonymize data by removing personally identifiable
            information, but this isn&apos;t enough to guarantee privacy on its
            own.
          </p>
        </div>
      </section>

      {/* Netflix Prize Section */}
      <section className="min-h-screen flex flex-col md:flex-row justify-between items-center p-6 md:p-12 md:mx-12">
        <div className="w-full md:w-1/2 prose prose-lg max-w-none mb-8 md:mb-0 text-center md:text-left">
          <p className="text-base md:text-xl">
            In 2006, Netflix created the{" "}
            <Link
              href="https://en.wikipedia.org/wiki/Netflix_Prize"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-accent cursor-pointer transition-colors underline decoration-dotted"
            >
              Netflix Prize
            </Link>
            , a competition to improve Netflix&apos;s movie recommendation
            algorithm. The dataset used in the competition contained 100 million
            ratings from 480,000 users on 17,770 movies, anonymized by removing
            personally identifiable information. One year later, using iMDB
            ratings as a reference, two researchers from UT Austin were able to{" "}
            <Link
              href="https://arxiv.org/abs/cs/0610105"
              target="_blank"
              rel="noopener noreferrer"
              className="hover:text-accent cursor-pointer transition-colors underline decoration-dotted"
            >
              deanonymize the 99% of the users in the dataset
            </Link>
            . This is why the rigor of DP is so important. Have you considered
            what someone could do with your personal watch history, ratings,
            browsing data, or worse?
          </p>
        </div>
        <div className="hidden md:flex w-full md:w-1/2 justify-center items-center">
          <div className="w-full h-full flex justify-center items-center">
            <div className="relative w-[50%] md:w-[70%]">
              <Image
                src={getPublicPath("netflix.svg")}
                alt="Netflix Logo"
                className="w-full h-auto"
                width={500}
                height={200}
                priority
              />
            </div>
          </div>
        </div>
      </section>

      {/* Interactive Visualization Section */}
      <section className="min-h-screen flex flex-col justify-center p-6 md:p-12">
        <HistogramViz />
      </section>

      {/* Logit Viz Section */}
      <section className="min-h-screen flex flex-col justify-center items-center p-6 md:p-12">
        <div className="prose prose-lg max-w-3xl mx-auto text-center">
          <p className="text-base md:text-xl">
            Differential Privacy can be used for more complex queries too, such
            as training a machine learning model. A data scientist may do this
            because they don&apos;t want to{" "}
            <button
              className="text-accent hover:text-primary-gray cursor-pointer transition-colors inline-flex items-center gap-1 group"
              onClick={() => setShowRevealDetails(!showRevealDetails)}
            >
              <span className="underline decoration-dotted">
                reveal if someone was in a training set
              </span>
              <span
                className={`inline-block transition-transform duration-200 ${
                  showRevealDetails ? "rotate-180" : ""
                }`}
              >
                ▼
              </span>
            </button>
            {". "}The question then becomes:
          </p>
          <p className="text-lg md:text-xl font-bold text-accent mt-4">
            When are differentially private methods practical and useful?
          </p>
          <p className="text-lg md:text-xl font-bold text-accent">
            How effective is differential privacy when applied in practice?
          </p>
          {/* Dropdown */}
          <div
            className={`mt-8 transition-all duration-500 overflow-hidden ${
              showRevealDetails
                ? "max-h-[500px] opacity-100"
                : "max-h-0 opacity-0"
            }`}
          >
            <div className="bg-primary-gray/5 p-6 rounded-lg border border-primary-gray/10">
              <h3 className="text-base md:text-lg font-semibold mb-4">
                Models can reveal what they were trained on!
              </h3>
              <p className="text-base md:text-lg mb-4">
                When ChatGPT 3.5 was released, trained on many datasets publicly
                and privately available. Clever prompters called
                &quot;Agents&quot; were able to gather SSN&apos;s for
                individuals which the model would produce with perfect accuracy!
                A differentially private algorithm guarantees that model outputs
                are not significantly different wether your SSN was in the
                training data or not, which means that the model would not
                reveal any information about the training data.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="min-h-screen flex flex-col justify-center p-6 md:p-12">
        <ConfidenceViz />
      </section>

      <section className="min-h-screen flex flex-col justify-center items-center p-6 md:p-12">
        <div className="prose prose-lg max-w-3xl mx-auto text-center">
          <p className="text-base md:text-xl">
            Overall, differential privacy is a powerful tool for protecting
            individual privacy while still allowing for useful data analysis.
            There exist many different ways to implement differential privacy,
            which requires asking questions like, &quot;Where should we add
            noise?&quot; or &quot;Are two neighboring datasets defined by a
            single entry, or a single user&apos;s worth of entries?&quot;
          </p>
          <p className="text-base md:text-xl mt-4">
            A lot of the research in DP has been finding the best ways to add
            noise while maintaining utilty and our following study focuses on
            how applicable it actually is in practice.
          </p>
        </div>
      </section>
    </div>
  );
}
