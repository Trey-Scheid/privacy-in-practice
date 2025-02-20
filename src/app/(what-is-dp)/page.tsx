"use client";

import { useRef } from "react";
import { useInView } from "react-intersection-observer";
import { SplitView } from "../../components/what-is-dp/split/SplitView";
import { NavBar } from "../../components/ui/NavBar";
import { CenteredView } from "../../components/what-is-dp/centered/CenteredView";
import { WhatIsDP } from "@/components/what-is-dp/what-is-dp";

export default function Home() {
  const { ref: titleRef, inView: isTitleVisible } = useInView({
    threshold: 0.5,
  });

  // Refs for sections
  const whatIsDPRef = useRef<HTMLElement>(null);

  const scrollToSection = (ref: React.RefObject<HTMLElement | null>) => {
    if (ref.current) {
      ref.current.scrollIntoView({ behavior: "smooth" });
    }
  };

  return (
    <div className="h-screen overflow-y-scroll snap-y snap-mandatory bg-primary-white text-primary-black">
      {/* Navigation Bar */}
      <NavBar
        isTitleVisible={isTitleVisible}
        scrollToSection={scrollToSection}
        whatIsDPRef={whatIsDPRef}
      />

      {/* What is DP Section */}
      <WhatIsDP
        SplitViewRef={whatIsDPRef}
        titleRef={titleRef}
        scrollToSection={scrollToSection}
      />
    </div>
  );
}
