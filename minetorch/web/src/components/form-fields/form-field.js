import Vue from 'vue'

export default {
  props: {
    help: String,
    label: String,
    name: String,
    formData: Object,
    default: null
  },
  mounted() {
    if (this.default) {
      Vue.set(this.formData, this.name, this.default)
    }
  }
}
