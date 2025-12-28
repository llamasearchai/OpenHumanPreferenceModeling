declare namespace GeoJSON {
  export interface Geometry {
    type: string;
    coordinates: unknown;
  }

  export interface Feature<G extends Geometry = Geometry, P = Record<string, unknown>> {
    type: 'Feature';
    geometry: G;
    properties: P;
    id?: string | number;
  }

  export interface FeatureCollection<
    G extends Geometry = Geometry,
    P = Record<string, unknown>
  > {
    type: 'FeatureCollection';
    features: Array<Feature<G, P>>;
  }
}

