import { Composition } from "remotion";
import { Demo } from "./Demo";
import timeline from "./timeline.json";
import short from "./timeline_short.json";

export const RemotionRoot: React.FC = () => {
  return (
    <>
      <Composition
        id="VeriCellDemo"
        component={Demo}
        durationInFrames={Math.ceil(timeline.total * 30)}
        fps={30}
        width={1920}
        height={1080}
      />
      <Composition
        id="VeriCellShort"
        component={Demo}
        durationInFrames={Math.ceil(short.total * 30)}
        fps={30}
        width={1920}
        height={1080}
        defaultProps={{ tl: short }}
      />
    </>
  );
};
