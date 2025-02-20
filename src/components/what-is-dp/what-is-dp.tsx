import { SplitView } from "./split/SplitView";
import { CenteredView } from "./centered/CenteredView";

interface WhatIsDPProps {
  SplitViewRef: React.RefObject<HTMLElement | null>;
  titleRef: (node?: Element | null | undefined) => void;
  scrollToSection: (ref: React.RefObject<HTMLElement | null>) => void;
}

export function WhatIsDP({
  SplitViewRef,
  titleRef,
  scrollToSection,
}: WhatIsDPProps) {
  return (
    <div>
      <SplitView
        SplitViewRef={SplitViewRef}
        titleRef={titleRef}
        scrollToSection={scrollToSection}
      />
      
      <CenteredView />
    </div>
  );
}
