<template>
  <div class="dynamic-form">
    <el-form ref="form" :model="formData" label-width="120px">
      <component v-for="field in schema" :is="field.type + '-field'" :key="field.label"
                 v-bind="field" :form-data="formData" />
      <el-form-item>
        <el-button type="primary" @click="onSubmit">{{ submitButton }}</el-button>
        <el-button @click="$emit('dynamic-form:cancel')">Cancel</el-button>
      </el-form-item>
    </el-form>
  </div>
</template>
<script>
import string from './form-fields/string'
import select from './form-fields/select'
import boolean from './form-fields/boolean'
import number from './form-fields/number'

export default {
  components: {
    'string-field': string,
    'select-field': select,
    'boolean-field': boolean,
    'number-field': number
  },
  props: {
    submitButton: {
      type: String,
      default: 'Create'
    },
    method: {
      type: String,
      default: 'post'
    },
    schema: {
      type: Array,
      default: function () {
        return []
      }
    },
    submitUrl: {
      type: String,
      default: ''
    },
    extraData: {
      type: Object,
      default: function() {
        return {}
      }
    }
  },
  data () {
    return {
      formData: {}
    }
  },
  methods: {
    async onSubmit() {
      const response = await this.ajax[this.method](this.submitUrl, Object.assign(this.extraData, this.formData))
      this.$emit('dynamic-form:success', response)
    }
  }
}
</script>
<style lang="scss">
.el-input {
  width: 300px;
}
</style>
