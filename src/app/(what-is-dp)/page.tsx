"use client";

import { useRef } from "react";
import { useInView } from "react-intersection-observer";
import { NavBar } from "../../components/ui/NavBar";
import { WhatIsDP } from "@/components/what-is-dp/what-is-dp";
import OurMethods from "@/components/our-methods/our-methods";
import Discussion from "@/components/discussion/discussion";

export default function Home() {
  const { ref: titleRef, inView: isTitleVisible } = useInView({
    threshold: 0.5,
  });

  // Refs for sections
  const topRef = useRef<HTMLDivElement>(null);
  const whatIsDPRef = useRef<HTMLElement>(null);
  const methodsRef = useRef<HTMLElement>(null);
  const discussionRef = useRef<HTMLElement>(null);

  const scrollToSection = (ref: React.RefObject<HTMLElement | null>) => {
    if (ref.current) {
      ref.current.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div className="h-screen overflow-y-scroll snap-y snap-mandatory bg-primary-white text-primary-black pt-16">
      {/* Navigation Bar */}
      <NavBar
        isTitleVisible={isTitleVisible}
        scrollToSection={scrollToSection}
        topRef={topRef}
        whatIsDPRef={whatIsDPRef}
        methodsRef={methodsRef}
        discussionRef={discussionRef}
      />

      {/* Top Section */}
      <div ref={topRef} className="h-0" />

      {/* What is DP Section */}
      <div className="snap-start" data-section="what-is-dp">
        <WhatIsDP
          SplitViewRef={whatIsDPRef}
          titleRef={titleRef}
          scrollToSection={scrollToSection}
          methodsRef={methodsRef}
          discussionRef={discussionRef}
        />
      </div>

      {/* Our Methods Section */}
      <div className="snap-start" data-section="methods">
        <OurMethods methodsRef={methodsRef} />
      </div>

      {/* Discussion Section */}
      <div className="snap-start" data-section="discussion">
        <Discussion discussionRef={discussionRef} />
      </div>
    </div>
  );
}
